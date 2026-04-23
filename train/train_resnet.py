# train_resnet.py
"""
ResNet 훈련 스크립트.

Residual connection 기반 CNN의 대표 아키텍처.
BatchNorm 사용 (bn/norm 키워드에 weight_decay 미적용).

지원 backbone (timm variant):
  resnet18     (~11.7M)
  resnet34     (~21.8M)
  resnet50     (~25.6M)
  resnet101    (~44.5M)
  resnet152    (~60.2M)

Ref: He et al., "Deep Residual Learning for Image Recognition"
     (CVPR 2016, arXiv:1512.03385)
"""
from dataclasses import dataclass
from typing import Optional

import timm

from train_base import BaseTrainConfig, setup_and_run


@dataclass
class TrainConfig(BaseTrainConfig):
    backbone_name: str = "resnet50"
    model_tag: str = "resnet50"
    pretrained: bool = False
    drop_rate: float = 0.0
    drop_path_rate: float = 0.05    # stochastic depth


# ResNet은 BatchNorm 기반이므로 bn/norm 키워드만 제외
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
        gpu_id=0,
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
        backbone_name="resnet50",
        model_tag="resnet50",
        pretrained=False,
        drop_rate=0.0,
        drop_path_rate=0.05,
    )
    main(cfg)
