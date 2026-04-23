# train_convnextv2.py
"""
ConvNeXtV2 훈련 스크립트.

GRN(Global Response Normalization) 레이어와 LayerNorm 사용.
norm/grn/gamma 파라미터에 weight_decay 미적용.

지원 backbone:
  convnextv2_atto    (~3.7M)
  convnextv2_femto   (~5.2M)
  convnextv2_pico    (~9.1M)
  convnextv2_nano    (~15.6M)
  convnextv2_tiny    (~28.6M)
  convnextv2_small   (~50M)
  convnextv2_base    (~89M)

Ref: Woo et al., "ConvNeXt V2: Co-designing and Scaling ConvNets
     with Masked Autoencoders" (arXiv:2301.00808)
"""
from dataclasses import dataclass
from typing import Optional

import timm

from train_base import BaseTrainConfig, setup_and_run


@dataclass
class TrainConfig(BaseTrainConfig):
    backbone_name: str = "convnextv2_femto"
    model_tag: str = "convnextv2_femto"
    pretrained: bool = False
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0


NO_DECAY_KEYWORDS = ("norm", "grn", "gamma")


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
        gpu_id=3,
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
        backbone_name="convnextv2_atto",
        model_tag="convnextv2_atto",
        pretrained=False,
        drop_rate=0.0,
        drop_path_rate=0.05,
    )
    main(cfg)
