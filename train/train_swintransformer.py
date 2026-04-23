# train_swin.py
"""
Swin Transformer 훈련 스크립트.

Window-based local attention + Shifted Window로 cross-window 정보 교환.
relative_position_bias_table, absolute_pos_embed에 weight_decay 미적용.

지원 backbone:
  swin_tiny_patch4_window7_224   (~28M, 4.5G FLOPs)
  swin_small_patch4_window7_224  (~50M)
  swin_base_patch4_window7_224   (~88M)

Ref: Liu et al., "Swin Transformer: Hierarchical Vision Transformer
     using Shifted Windows" (arXiv:2103.14030)
"""
from dataclasses import dataclass
from typing import Optional

import timm

from train_base import BaseTrainConfig, setup_and_run


@dataclass
class TrainConfig(BaseTrainConfig):
    backbone_name: str = "swin_tiny_patch4_window7_224"
    model_tag: str = "swin_tiny"
    pretrained: bool = False
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0


NO_DECAY_KEYWORDS = ("norm", "relative_position_bias_table", "absolute_pos_embed")


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
        gpu_id=2,
        data_dir="/home/dongbeen/ML/Paper/AnchorGViT/dataset_split",
        img_size=224,
        batch_size=64,
        epochs=300,
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
        backbone_name="swin_tiny_patch4_window7_224",
        model_tag="swin_tiny",
        pretrained=False,
        drop_rate=0.0,
        drop_path_rate=0.0,
    )
    main(cfg)
