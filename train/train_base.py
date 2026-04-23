import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import copy
import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    balanced_accuracy_score,
    cohen_kappa_score,
)
from sklearn.model_selection import StratifiedKFold

try:
    import wandb
except ImportError:
    wandb = None

from data.dataset import CanineOcularDataset
from utils.callbacks import EarlyStopping

@dataclass
class BaseTrainConfig:
    gpu_id: int = 0
    data_dir: str = "/home/dongbeen/ML/Paper/PSAN/dataset_split"
    img_size: int = 224
    batch_size: int = 64
    epochs: int = 200
    lr: float = 1e-4
    weight_decay: float = 1e-3
    num_workers: int = 8
    seed: int = 42
    num_classes: int = 11
    deterministic: bool = True

    early_stopping_patience: int = 20
    warmup_epochs: int = 5
    ema_decay: float = 0.999
    ema_warmup_epochs: int = 20

    backbone_name: str = ""
    model_tag: str = ""
    pretrained: bool = False
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0

def correct_count_from_logits(logits, y):
    pred = logits.argmax(dim=1)
    return (pred == y).float().sum().item()


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ModelEMA:
    def __init__(self, model, decay=0.999, device=None):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        if device is not None:
            self.ema.to(device)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        esd = self.ema.state_dict()
        for k, v in esd.items():
            if v.dtype.is_floating_point:
                v.copy_(v * self.decay + msd[k].detach() * (1.0 - self.decay))
            else:
                v.copy_(msd[k])

def build_param_groups(model, weight_decay, no_decay_keywords=()):
    """Weight decay / no-decay 파라미터 그룹 분리.

    p.ndim <= 1, ".bias"는 항상 no-decay.
    추가로 이름에 no_decay_keywords의 키워드가 포함되면 no-decay.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (
            p.ndim <= 1
            or name.endswith(".bias")
            or any(kw in name for kw in no_decay_keywords)
        ):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_scheduler(optimizer, warmup_epochs, total_epochs):
    if warmup_epochs > 0 and total_epochs > warmup_epochs:
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)
            progress = float(epoch - warmup_epochs) / float(
                total_epochs - warmup_epochs
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs
        )

def build_transforms(img_size):
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    tfm_train = transforms.Compose(
        [
            transforms.Resize(
                (img_size, img_size)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.10, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]
    )
    tfm_val = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]
    )
    return tfm_train, tfm_val
    
def build_dataloaders(cfg, tfm_train, tfm_val):
    print("[Info] Using MulticlassEyeDataset")
    train_ds = CanineOcularDataset(
        cfg.data_dir, split="train", img_size=cfg.img_size, transform=tfm_train
    )
    val_ds = CanineOcularDataset(
        cfg.data_dir, split="val", img_size=cfg.img_size, transform=tfm_val
    )

    y_train = [y for _, y in train_ds.samples]
    class_counts = np.bincount(y_train, minlength=cfg.num_classes)

    weight_per_class = 1.0 / np.sqrt(np.maximum(class_counts, 1))
    sample_weights = torch.tensor(
        [weight_per_class[y] for y in y_train], dtype=torch.float32
    )
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )

    print(f"Class counts: {class_counts}")
    return train_ds, val_ds, train_loader, val_loader, class_counts

def _build_combined_dataset(cfg, transform, splits=("train", "val")):
    """train/val 폴더의 샘플을 하나의 MulticlassEyeDataset로 병합.

    모든 split의 class 목록이 동일해야 한다(같은 클래스 → 같은 index).
    """
    assert len(splits) >= 1, "splits must be non-empty"
    ds = MulticlassEyeDataset(
        cfg.data_dir, split=splits[0], img_size=cfg.img_size, transform=transform
    )
    for s in splits[1:]:
        extra = MulticlassEyeDataset(
            cfg.data_dir, split=s, img_size=cfg.img_size, transform=transform
        )
        assert extra.classes == ds.classes, (
            f"Class mismatch between splits {splits[0]} and {s}: "
            f"{ds.classes} vs {extra.classes}"
        )
        ds.samples.extend(extra.samples)
    return ds


def build_cv_dataloaders(
    cfg,
    fold_idx,
    n_folds,
    fold_seed,
    tfm_train,
    tfm_val,
    splits=("train", "val"),
):

    full_train = _build_combined_dataset(cfg, tfm_train, splits=splits)
    full_val = _build_combined_dataset(cfg, tfm_val, splits=splits)
    assert len(full_train.samples) == len(full_val.samples)

    labels = np.array([y for _, y in full_train.samples])
    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=fold_seed
    )
    splits_list = list(skf.split(np.zeros(len(labels)), labels))
    train_idx, val_idx = splits_list[fold_idx]

    train_ds = Subset(full_train, train_idx)
    val_ds = Subset(full_val, val_idx)
  
    train_ds.classes = full_train.classes
    val_ds.classes = full_val.classes

    y_train = labels[train_idx]
    class_counts = np.bincount(y_train, minlength=cfg.num_classes)

    weight_per_class = 1.0 / np.sqrt(np.maximum(class_counts, 1))
    sample_weights = torch.tensor(
        [weight_per_class[y] for y in y_train], dtype=torch.float32
    )
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )

    print(
        f"[CV Fold {fold_idx + 1}/{n_folds}] "
        f"train={len(train_idx)} val={len(val_idx)} "
        f"(total={len(labels)}, seed={fold_seed})"
    )
    print(f"[CV Fold {fold_idx + 1}] Class counts (train): {class_counts}")
    return train_ds, val_ds, train_loader, val_loader, class_counts

def train_one_epoch(model, loader, optimizer, device, ema=None):
    """표준 train loop (cross-entropy). Returns (loss, acc)."""
    model.train()

    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if ema is not None:
            ema.update(model)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += correct_count_from_logits(logits, y)
        n += bs

    return total_loss / n, total_acc / n * 100.0


@torch.no_grad()
def validate(model, loader, device, num_classes):
    model.eval()

    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    all_preds = []
    all_labels = []
    all_probs = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)

        logits = logits.float()
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_correct += (preds == y).float().sum().item()
        total_samples += bs

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(y.cpu().numpy().tolist())

        if num_classes > 2:
            all_probs.extend(probs.cpu().numpy().tolist())
        else:
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples * 100.0

    metrics = {}
    try:
        if num_classes > 2:
            macro_auc = roc_auc_score(
                all_labels,
                np.array(all_probs),
                multi_class="ovr",
                average="macro",
            )
        else:
            macro_auc = roc_auc_score(all_labels, np.array(all_probs))
        metrics["auc"] = macro_auc
        metrics["macro_auc"] = macro_auc
    except Exception:
        metrics["auc"] = 0.0
        metrics["macro_auc"] = 0.0

    report = classification_report(
        all_labels,
        all_preds,
        output_dict=True,
        zero_division=0,
    )
    metrics["f1"] = report["weighted avg"]["f1-score"]
    metrics["macro_f1"] = report["macro avg"]["f1-score"]
    metrics["bal_acc"] = balanced_accuracy_score(all_labels, all_preds)
    metrics["kappa"] = cohen_kappa_score(all_labels, all_preds)
    metrics["predictions"] = np.array(all_probs)
    metrics["targets"] = np.array(all_labels)

    return avg_loss, avg_acc, metrics

def run_training(
    cfg,
    model,
    device,
    no_decay_keywords=(),
    train_epoch_fn=None,
    loaders=None,
    fold_tag="",
    wandb_group=None,
    validate_fn=None,
):

    output_tag = cfg.model_tag + fold_tag

    if wandb is None:
        print("wandb is not installed; skipping W&B logging.")
    else:
        wandb_project = os.getenv("WANDB_PROJECT", "anchorgvit-5090")
        wandb_run_name = os.getenv("WANDB_RUN_NAME", output_tag)
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            group=wandb_group,
            config={
                **asdict(cfg),
                "device": str(device),
                "fold_tag": fold_tag,
            },
            reinit=True,
        )

    if loaders is None:
        tfm_train, tfm_val = build_transforms(cfg.img_size)
        train_ds, val_ds, train_loader, val_loader, class_counts = (
            build_dataloaders(cfg, tfm_train, tfm_val)
        )
    else:
        train_ds, val_ds, train_loader, val_loader, class_counts = loaders

    if wandb is not None and wandb.run is not None:
        wandb.config.update(
            {"class_counts": class_counts.tolist()}, allow_val_change=True
        )

    for param in model.parameters():
        param.data = param.data.contiguous()

    param_groups = build_param_groups(model, cfg.weight_decay, no_decay_keywords)
    n_decay = sum(p.numel() for p in param_groups[0]["params"])
    n_no_decay = sum(p.numel() for p in param_groups[1]["params"])
    print(
        f"[Optim] AdamW param groups | decay={n_decay:,} "
        f"no_decay={n_no_decay:,} (wd={cfg.weight_decay})"
    )

    optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr)
    scheduler = build_scheduler(optimizer, cfg.warmup_epochs, cfg.epochs)

    ema = (
        ModelEMA(model, decay=cfg.ema_decay, device=device)
        if cfg.ema_decay > 0.0
        else None
    )

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())

    print("\n[Model Statistics]")
    print(f"  - Backbone: {cfg.backbone_name}")
    print(f"  - Total Params: {total_params:,} ({total_params / 1e6:.2f} M)")
    print(f"  - Trainable Params: {trainable_params:,}")

    if device.type == "cuda":
        print(
            f"  - GPU Memory Allocated: "
            f"{torch.cuda.memory_allocated(device) / 1024**2:.2f} MB"
        )
        print(
            f"  - GPU Memory Reserved:  "
            f"{torch.cuda.memory_reserved(device) / 1024**2:.2f} MB"
        )

    _validate = validate_fn if validate_fn is not None else validate

    early_stopping = EarlyStopping(
        patience=cfg.early_stopping_patience, rank=0, mode="max"
    )

    best_val_f1 = 0.0
    results = []

    print(f"\n=== {output_tag} | {cfg.epochs} epochs ===")

    for ep in range(1, cfg.epochs + 1):
        ep_start = time.time()

        if train_epoch_fn is not None:
            tr_loss, tr_acc, extra_log = train_epoch_fn(
                model, train_loader, optimizer, device, ema
            )
        else:
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, optimizer, device, ema=ema,
            )
            extra_log = {}

        if ema is not None and ep > cfg.ema_warmup_epochs:
            eval_model = ema.ema
            eval_source = "ema"
        else:
            eval_model = model
            eval_source = "model"

        va_loss, va_acc, va_metrics = _validate(                     
            eval_model, val_loader, device, cfg.num_classes,
        )

        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        ep_time = time.time() - ep_start
        f1 = va_metrics.get("f1", 0.0)
        macro_f1 = va_metrics.get("macro_f1", 0.0)
        auc = va_metrics.get("auc", 0.0)
        macro_auc = va_metrics.get("macro_auc", 0.0)
        bal_acc = va_metrics.get("bal_acc", 0.0)
        kappa = va_metrics.get("kappa", 0.0)

        def build_ckpt():
            return {
                "epoch": ep,
                "model": model.state_dict(),
                "ema": ema.ema.state_dict() if ema is not None else None,
                "eval_model": eval_model.state_dict(),
                "eval_source": eval_source,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": asdict(cfg),
                "best_macro_f1": max(best_val_f1, macro_f1),
                "val_metrics": {
                    "loss": va_loss,
                    "acc": va_acc,
                    "f1": f1,
                    "macro_f1": macro_f1,
                    "macro_auc": macro_auc,
                    "bal_acc": bal_acc,
                    "kappa": kappa,
                },
                "rng_state": {
                    "torch": torch.get_rng_state(),
                    "cuda": (
                        torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else None
                    ),
                    "numpy": np.random.get_state(),
                    "python": random.getstate(),
                },
            }

        is_best = macro_f1 > best_val_f1
        if is_best:
            best_val_f1 = macro_f1
            torch.save(build_ckpt(), f"best_{output_tag}.pth")

        torch.save(build_ckpt(), f"last_{output_tag}.pth")

        best_mark = " *" if is_best else ""
        print(
            f"[Ep {ep:03d}/{cfg.epochs}]{best_mark} "
            f"tloss={tr_loss:.4f} tacc={tr_acc:.2f}% | "
            f"vloss={va_loss:.4f} vacc={va_acc:.2f}% ({eval_source}) | "
            f"f1={f1:.4f} macro_f1={macro_f1:.4f} macro_auc={macro_auc:.4f} "
            f"bal_acc={bal_acc:.4f} kappa={kappa:.4f} | "
            f"lr={lr:.2e} time={ep_time:.1f}s | "
            f"best_macro_f1={best_val_f1:.4f} "
            f"patience={early_stopping.counter}/{cfg.early_stopping_patience}"
        )

        if wandb is not None and wandb.run is not None:
            log_dict = {
                "train/loss": tr_loss,
                "train/acc": tr_acc,
                "val/loss": va_loss,
                "val/acc": va_acc,
                "val/f1": f1,
                "val/macro_f1": macro_f1,
                "val/auc": auc,
                "val/macro_auc": macro_auc,
                "best/val_macro_f1": best_val_f1,
                "lr": lr,
                "epoch_time_sec": ep_time,
            }
            for k in ("s1_acc", "s1_macro_f1", "s2_acc", "s2_macro_f1",
                      "cascade_macro_f1"):
                if k in va_metrics:
                    log_dict[f"val/{k}"] = va_metrics[k]
            log_dict.update(extra_log)
            wandb.log(log_dict, step=ep)

            if ep % 5 == 0:
                preds_np = va_metrics["predictions"]
                if preds_np.ndim > 1:
                    y_pred_classes = np.argmax(preds_np, axis=1)
                else:
                    y_pred_classes = (preds_np > 0.5).astype(int)

                wandb.log(
                    {
                        "conf_mat": wandb.plot.confusion_matrix(
                            probs=None,
                            y_true=va_metrics["targets"],
                            preds=y_pred_classes,
                            class_names=getattr(train_ds, "classes", None),
                        )
                    },
                    step=ep,
                )

        results.append(
            {
                "epoch": ep,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": va_loss,
                "val_acc": va_acc,
                "val_f1": f1,
                "val_macro_f1": macro_f1,
                "val_auc": auc,
                "val_macro_auc": macro_auc,
                "val_bal_acc": bal_acc,
                "val_kappa": kappa,
                "best_val_f1_so_far": best_val_f1,
                "epoch_time_sec": ep_time,
            }
        )

        early_stopping(macro_f1)
        if early_stopping.early_stop:
            print(f"early stopped @ ep {ep}")
            break

    df = pd.DataFrame(results)
    csv_name = f"{output_tag}_results_detailed.csv"
    df.to_csv(csv_name, index=False)

    print(f"\nSaved: best_{output_tag}.pth")
    print(f"Saved: last_{output_tag}.pth")
    print(f"Saved: {csv_name}")

    if wandb is not None and wandb.run is not None:
        wandb.finish()

    return best_val_f1

def setup_and_run(cfg, model_fn, no_decay_keywords=(), train_epoch_fn=None, validate_fn=None):
    set_seed(cfg.seed, deterministic=cfg.deterministic)
    print(
        f"[cuDNN] deterministic={torch.backends.cudnn.deterministic} "
        f"benchmark={torch.backends.cudnn.benchmark}"
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.gpu_id)
        device = torch.device(f"cuda:{cfg.gpu_id}")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    task_type = "Multiclass" if cfg.num_classes > 2 else "Binary"
    print(
        f"Model: {cfg.backbone_name} "
        f"({'pretrained' if cfg.pretrained else 'no pretrain'}) | "
        f"{task_type} Classification"
    )

    model = model_fn(cfg, device)

    run_training(
        cfg,
        model,
        device,
        no_decay_keywords=no_decay_keywords,
        train_epoch_fn=train_epoch_fn,
        validate_fn=validate_fn, 
    )
