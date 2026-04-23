# train_gfnet_tiny.py
"""
GFNet-Ti 훈련 스크립트.

GFNet = 학습 가능한 spectral filter (FFT → filter → iFFT) 기반 token mixer.
KerrViT의 가장 직접적인 baseline (KerrViT에서 γ=0이면 GFNet으로 환원).

아키텍처 스펙은 KerrViT-Tiny와 동일하게 맞춤:
  patch_size=16, embed_dim=192, depth=12, mlp_ratio=4.0, grid=14×14
  → 약 4.2M 파라미터 (KerrViT-Tiny 4.23M과 거의 동일)

Weight decay 제외 대상:
  - LayerNorm (norm)
  - filter_real / filter_imag : spectral filter 파라미터는 attention_bias와
    유사한 성격이므로 KerrViT 논문 실험과 일관되게 decay 미적용 권장.

Ref: Rao et al., "Global Filter Networks for Image Classification"
     (NeurIPS 2021, arXiv:2107.00645)
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataclasses import dataclass
from typing import Optional

from models.gfnet import gfnet_tiny

from train_base import BaseTrainConfig, setup_and_run


@dataclass
class TrainConfig(BaseTrainConfig):
    model_tag: str = "gfnet_tiny"
    drop_rate: float = 0.0
    drop_path_rate: float = 0.1     # GFNet 원논문 Ti 설정


# GFNet은 Conv stem의 BN이 없고 LayerNorm만 사용.
# filter_real / filter_imag는 spectral filter로, weight decay에서 제외하는 것이
# KerrViT의 attention_bias 제외 정책과 일관됨.
NO_DECAY_KEYWORDS = ("norm", "filter_real", "filter_imag")


def build_model(cfg, device):
    return gfnet_tiny(
        num_classes=cfg.num_classes,
        img_size=cfg.img_size,
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
        model_tag="gfnet_tiny",
        drop_rate=0.0,
        drop_path_rate=0.1,
    )
    main(cfg)