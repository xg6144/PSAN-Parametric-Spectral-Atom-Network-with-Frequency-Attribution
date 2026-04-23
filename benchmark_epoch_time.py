"""모든 train/train_*.py 모델의 1 에포크 훈련 시간을 10 에포크 동안 측정하여
평균 낸 뒤 JSON 으로 저장하는 벤치마크 스크립트.

- 가중치(.pth)는 저장하지 않는다.
- Validation / EMA / wandb / 스케줄러 / early-stopping 은 실행하지 않는다.
  (순수 훈련 루프 시간만 측정)
- 각 스크립트의 `__main__` 블록 하이퍼파라미터를 재현하되, epochs=10 으로 덮어쓴다.

사용법:
    python benchmark_epoch_time.py                    # 전체 실행 → JSON 저장
    python benchmark_epoch_time.py --only deit_tiny   # 특정 모델만
    python benchmark_epoch_time.py --gpu-id 1         # 다른 GPU
"""

import argparse
import gc
import importlib
import json
import os
import statistics
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_DIR = PROJECT_ROOT / "train"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TRAIN_DIR))

from train_base import (  # noqa: E402
    build_dataloaders,
    build_param_groups,
    build_transforms,
    correct_count_from_logits,
    set_seed,
)


# ---------------------------------------------------------------------------
# 모델별 하이퍼파라미터 (각 train_*.py 의 __main__ 블록과 일치)
# ---------------------------------------------------------------------------
# `common` 는 전 모델 공통값, `extra` 는 모델별 추가 오버라이드이다.
MODEL_SPECS = [
    {
        "label": "deit_tiny",
        "module": "train_deit",
        "config_attr": "TrainConfig",
        "no_decay_attr": "NO_DECAY_KEYWORDS",
        "extra": dict(
            backbone_name="deit_tiny_patch16_224",
            model_tag="deit_tiny",
            pretrained=False,
            drop_rate=0.0,
            drop_path_rate=0.05,
            warmup_epochs=15,
        ),
    },
    {
        "label": "resnet50",
        "module": "train_resnet",
        "config_attr": "TrainConfig",
        "no_decay_attr": "NO_DECAY_KEYWORDS",
        "extra": dict(
            backbone_name="resnet50",
            model_tag="resnet50",
            pretrained=False,
            drop_rate=0.0,
            drop_path_rate=0.05,
            warmup_epochs=15,
        ),
    },
    {
        "label": "efficientnet_b0",
        "module": "train_efficientnet",
        "config_attr": "TrainConfig",
        "no_decay_attr": "NO_DECAY_KEYWORDS",
        "extra": dict(
            backbone_name="efficientnet_b0",
            model_tag="efficientnet_b0",
            pretrained=False,
            drop_rate=0.0,
            drop_path_rate=0.05,
            warmup_epochs=15,
        ),
    },
    {
        "label": "convnextv2_atto",
        "module": "train_convnextv2",
        "config_attr": "TrainConfig",
        "no_decay_attr": "NO_DECAY_KEYWORDS",
        "extra": dict(
            backbone_name="convnextv2_atto",
            model_tag="convnextv2_atto",
            pretrained=False,
            drop_rate=0.0,
            drop_path_rate=0.05,
            warmup_epochs=15,
        ),
    },
    {
        "label": "swin_tiny",
        "module": "train_swintransformer",
        "config_attr": "TrainConfig",
        "no_decay_attr": "NO_DECAY_KEYWORDS",
        "extra": dict(
            backbone_name="swin_tiny_patch4_window7_224",
            model_tag="swin_tiny",
            pretrained=False,
            drop_rate=0.0,
            drop_path_rate=0.0,
            warmup_epochs=15,
        ),
    },
    {
        "label": "mlp_mixer_tiny",
        "module": "train_mlp_mixer",
        "config_attr": "MixerConfig",
        "no_decay_attr": "MIXER_NO_DECAY_KEYWORDS",
        "extra": dict(
            backbone_name="MLP-Mixer-Tiny",
            model_tag="mlp_mixer_tiny",
            model_variant="mlp_mixer_tiny",
            drop_path_rate=0.1,
            use_checkpoint=False,
            warmup_epochs=15,
        ),
    },
    {
        "label": "gfnet_tiny",
        "module": "train_gfnet",
        "config_attr": "TrainConfig",
        "no_decay_attr": "NO_DECAY_KEYWORDS",
        "extra": dict(
            model_tag="gfnet_tiny",
            drop_rate=0.0,
            drop_path_rate=0.1,
            warmup_epochs=15,
        ),
    },
    {
        "label": "afno_tiny",
        "module": "train_afno",
        "config_attr": "TrainConfig",
        "no_decay_attr": "NO_DECAY_KEYWORDS",
        "extra": dict(
            backbone_name="AFNO-Tiny",
            model_tag="afno_tiny",
            pretrained=False,
            drop_rate=0.0,
            drop_path_rate=0.1,
            model_variant="afno_tiny",
            sparsity_thresh=0.01,
            hard_thresh_frac=1.0,
            use_checkpoint=False,
            lr=5e-4,
            weight_decay=0.05,
            warmup_epochs=15,
        ),
    },
    {
        "label": "pspn_gfnet_ti_M16_gabor_morlet",
        "module": "train_pspn_gfnet",
        "config_attr": "TrainConfig",
        "no_decay_attr": "NO_DECAY_KEYWORDS",
        "extra": dict(
            model_tag="pspn_gfnet_ti_M16_gabor_morlet",
            model="pspn_gfnet_ti",
            drop_rate=0.0,
            drop_path_rate=0.1,
            atom_count=16,
            atom="gabor",
            init="morlet",
            isotropic=False,
            no_phase=False,
            bins_dropout=0.0,
            pspn_pretrained="",
            warmup_epochs=15,
        ),
    },
]


# ---------------------------------------------------------------------------
# 순수 훈련 루프 (가중치 저장 / val / ema / wandb 없음)
# ---------------------------------------------------------------------------
def train_one_epoch_bare(model, loader, optimizer, device):
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

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += correct_count_from_logits(logits, y)
        n += bs
    return total_loss / n, total_acc / n * 100.0


def benchmark_one(spec: dict, base_cfg: dict, device: torch.device) -> dict:
    label = spec["label"]
    print(f"\n{'=' * 70}\n[Benchmark] {label}\n{'=' * 70}")

    module = importlib.import_module(spec["module"])
    ConfigCls = getattr(module, spec["config_attr"])
    build_model = getattr(module, "build_model")
    no_decay_keywords = getattr(module, spec["no_decay_attr"], ())

    cfg_kwargs = dict(base_cfg)
    cfg_kwargs.update(spec["extra"])
    # 벤치마크 강제 오버라이드
    cfg_kwargs["epochs"] = base_cfg["epochs"]
    cfg_kwargs["gpu_id"] = base_cfg["gpu_id"]
    # 미사용 기능 off — 빠른 데이터 로더 초기화
    cfg_kwargs.setdefault("early_stopping_patience", 9999)

    cfg = ConfigCls(**cfg_kwargs)

    set_seed(cfg.seed, deterministic=cfg.deterministic)

    model = build_model(cfg, device)
    model = model.to(device)
    for p in model.parameters():
        p.data = p.data.contiguous()

    param_groups = build_param_groups(model, cfg.weight_decay, no_decay_keywords)
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr)

    tfm_train, tfm_val = build_transforms(cfg.img_size)
    train_ds, val_ds, train_loader, val_loader, class_counts = build_dataloaders(
        cfg, tfm_train, tfm_val
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"  params total={total_params:,} trainable={trainable_params:,} | "
        f"device={device} batch={cfg.batch_size} img={cfg.img_size} "
        f"workers={cfg.num_workers}"
    )

    per_epoch_seconds: list[float] = []
    per_epoch_loss: list[float] = []
    per_epoch_acc: list[float] = []

    for ep in range(1, cfg.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        tr_loss, tr_acc = train_one_epoch_bare(
            model, train_loader, optimizer, device
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0

        per_epoch_seconds.append(dt)
        per_epoch_loss.append(tr_loss)
        per_epoch_acc.append(tr_acc)
        print(
            f"  [Ep {ep:02d}/{cfg.epochs}] "
            f"time={dt:.2f}s  tloss={tr_loss:.4f}  tacc={tr_acc:.2f}%"
        )

    avg = statistics.fmean(per_epoch_seconds)
    avg_excl_first = (
        statistics.fmean(per_epoch_seconds[1:]) if len(per_epoch_seconds) > 1 else avg
    )
    med = statistics.median(per_epoch_seconds)
    stdev = (
        statistics.pstdev(per_epoch_seconds) if len(per_epoch_seconds) > 1 else 0.0
    )

    result = {
        "label": label,
        "module": spec["module"],
        "config": asdict(cfg),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "epochs_measured": cfg.epochs,
        "per_epoch_seconds": per_epoch_seconds,
        "per_epoch_train_loss": per_epoch_loss,
        "per_epoch_train_acc": per_epoch_acc,
        "avg_epoch_seconds": avg,
        "avg_epoch_seconds_excl_first": avg_excl_first,
        "median_epoch_seconds": med,
        "stdev_epoch_seconds": stdev,
        "total_seconds": sum(per_epoch_seconds),
    }

    print(
        f"  → avg={avg:.2f}s  avg(excl 1st)={avg_excl_first:.2f}s  "
        f"median={med:.2f}s  std={stdev:.2f}s"
    )

    # 명시적 메모리 해제
    del model, optimizer, param_groups
    del train_loader, val_loader, train_ds, val_ds
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/dongbeen/ML/Paper/AnchorGViT/dataset_split",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "epoch_time_benchmark.json"),
    )
    parser.add_argument(
        "--only",
        type=str,
        nargs="+",
        default=None,
        help="특정 label 만 실행 (예: --only deit_tiny resnet50)",
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")

    base_cfg = dict(
        gpu_id=args.gpu_id,
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=1e-4,
        weight_decay=1e-3,
        num_workers=args.num_workers,
        seed=42,
        num_classes=11,
        deterministic=True,
        early_stopping_patience=9999,
        warmup_epochs=15,
        ema_decay=0.999,
        ema_warmup_epochs=20,
    )

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"CUDA device name: {torch.cuda.get_device_name(device)}")

    specs = MODEL_SPECS
    if args.only:
        wanted = set(args.only)
        specs = [s for s in MODEL_SPECS if s["label"] in wanted]
        if not specs:
            print(f"[Error] --only 에 일치하는 모델이 없다: {args.only}")
            print(f"가능한 label: {[s['label'] for s in MODEL_SPECS]}")
            sys.exit(1)

    results = []
    failures = []
    for spec in specs:
        try:
            res = benchmark_one(spec, base_cfg, device)
            results.append(res)
        except Exception as e:
            print(f"[Error] {spec['label']} 실패: {e}")
            traceback.print_exc()
            failures.append({"label": spec["label"], "error": str(e)})
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary = [
        {
            "label": r["label"],
            "avg_epoch_seconds": r["avg_epoch_seconds"],
            "avg_epoch_seconds_excl_first": r["avg_epoch_seconds_excl_first"],
            "median_epoch_seconds": r["median_epoch_seconds"],
            "total_params": r["total_params"],
        }
        for r in results
    ]

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "epochs_measured": args.epochs,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "num_workers": args.num_workers,
        "data_dir": args.data_dir,
        "summary": summary,
        "results": results,
        "failures": failures,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}\n[Saved] {out_path}\n{'=' * 70}")
    print("Summary (avg seconds per epoch):")
    for s in summary:
        print(
            f"  {s['label']:<40s}  "
            f"avg={s['avg_epoch_seconds']:.2f}s  "
            f"avg(excl 1st)={s['avg_epoch_seconds_excl_first']:.2f}s  "
            f"params={s['total_params']:,}"
        )
    if failures:
        print("\nFailed models:")
        for f in failures:
            print(f"  {f['label']}: {f['error']}")


if __name__ == "__main__":
    main()
