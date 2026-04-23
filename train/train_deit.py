# train_deit_tiny.py
from dataclasses import dataclass
from typing import Optional

import timm

from train_base import BaseTrainConfig, setup_and_run


@dataclass
class TrainConfig(BaseTrainConfig):
    backbone_name: str = "deit_tiny_patch16_224"
    model_tag: str = "deit_tiny"
    pretrained: bool = False
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0


NO_DECAY_KEYWORDS = ("pos_embed", "cls_token", "dist_token")


def build_model(cfg, device):
    return timm.create_model(
        cfg.backbone_name,
        pretrained=cfg.pretrained,
        num_classes=cfg.num_classes,
        drop_rate=cfg.drop_rate,
        drop_path_rate=cfg.drop_path_rate,
    ).to(device)


def main(cfg: Optional[TrainConfig] = None):
    cfg = cfg or TrainConfig()
    setup_and_run(cfg, build_model, no_decay_keywords=NO_DECAY_KEYWORDS)


if __name__ == "__main__":
    cfg = TrainConfig(
        gpu_id=0,
        data_dir="/home/dongbeen/ML/Paper/AnchorGViT/dataset_split",
        img_size=224,
        batch_size=64,
        epochs=200,
        lr=1e-4,
        weight_decay=1e-3,
        num_workers=8,
        seed=42,
        num_classes=11,
        deterministic=True,
        early_stopping_patience=20,
        warmup_epochs=15,
        ema_decay=0.999,
        ema_warmup_epochs=20,
        backbone_name="deit_tiny_patch16_224",
        model_tag="deit_tiny",
        pretrained=False,
        drop_rate=0.0,
        drop_path_rate=0.05,
    )
    main(cfg)
