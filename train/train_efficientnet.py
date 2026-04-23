# train_efficientnet_b0.py
"""
EfficientNet-B0 훈련 스크립트.

Compound scaling (depth/width/resolution) 기반 MBConv 계열 CNN.
BatchNorm에는 weight_decay 미적용 (일반적인 CNN 훈련 관례).

지원 backbone (timm variant):
  efficientnet_b0   (~5.3M)
  efficientnet_b1   (~7.8M)
  efficientnet_b2   (~9.2M)
  efficientnet_b3   (~12M)

Ref: Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional
     Neural Networks" (ICML 2019, arXiv:1905.11946)
"""
from dataclasses import dataclass
from typing import Optional

import timm

from train_base import BaseTrainConfig, setup_and_run


@dataclass
class TrainConfig(BaseTrainConfig):
    backbone_name: str = "efficientnet_b0"
    model_tag: str = "efficientnet_b0"
    pretrained: bool = False
    drop_rate: float = 0.0          # EfficientNet 원논문 기본값
    drop_path_rate: float = 0.05     # stochastic depth


# EfficientNet은 BatchNorm 기반이므로 bn/norm 키워드만 제외
NO_DECAY_KEYWORDS = ("bn", "norm")


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
        gpu_id=1,
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
        backbone_name="efficientnet_b0",
        model_tag="efficientnet_b0",
        pretrained=False,
        drop_rate=0.0,
        drop_path_rate=0.05,
    )
    main(cfg)