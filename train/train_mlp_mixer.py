"""
train_mlp_mixer.py — MLP-Mixer-Tiny training on the canine ocular dataset.

Uses shared train_base infrastructure so comparisons against
PSPN-GFNet-Ti / GFNet-tiny / AFNO-Tiny / other baselines are run
under identical conditions (transforms, WeightedRandomSampler, cosine
schedule, EMA, early stopping, wandb).

Reference: Tolstikhin et al., "MLP-Mixer: An all-MLP Architecture for
Vision", NeurIPS 2021.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataclasses import dataclass
from typing import Optional

from models import mlp_mixer
from train_base import BaseTrainConfig, setup_and_run


# ======================================================================
#  HYPERPARAMETERS — edit this block to change a run.
# ======================================================================
@dataclass
class MixerConfig(BaseTrainConfig):
    # ---- Bookkeeping ----
    backbone_name: str = "MLP-Mixer-Tiny"
    model_tag: str = "mlp_mixer_tiny"

    # ---- Variant selection ----
    model_variant: str = "mlp_mixer_tiny"   # tiny | small | base

    # ---- Regularization ----
    drop_path_rate: float = 0.1

    # ---- Memory / compute ----
    use_checkpoint: bool = False


def build_model(cfg: MixerConfig, device):
    factory = getattr(mlp_mixer, cfg.model_variant)
    model = factory(
        num_classes=cfg.num_classes,
        drop_path_rate=cfg.drop_path_rate,
        use_checkpoint=cfg.use_checkpoint,
    )
    model = model.to(device)

    pc = mlp_mixer.count_params(model)
    grid = model.patch_embed.grid_h
    print(
        f"[MLP-Mixer] variant={cfg.model_variant}  "
        f"patch={model.patch_embed.patch_size}  grid={grid}x{grid}  "
        f"tokens={model.num_tokens}  "
        f"params total={pc['total']:,}  "
        f"token_mix={pc['token_mix']:,}  channel_mix={pc['channel_mix']:,}"
    )
    return model


# ----------------------------------------------------------------------
# No-decay notes:
#   build_param_groups already sends ndim <= 1 parameters (biases,
#   LayerNorm weights) to the no-decay group.  MLP-Mixer has no
#   special geometry parameters, so no extra keywords are needed.
# ----------------------------------------------------------------------
MIXER_NO_DECAY_KEYWORDS = ()


def main(cfg: Optional[MixerConfig] = None):
    cfg = cfg or MixerConfig()
    setup_and_run(cfg, build_model, no_decay_keywords=MIXER_NO_DECAY_KEYWORDS)


if __name__ == "__main__":
    cfg = MixerConfig(
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
        backbone_name="MLP-Mixer-Tiny",
        model_tag="mlp_mixer_tiny",
        model_variant="mlp_mixer_tiny",
        drop_path_rate=0.1,
        use_checkpoint=False,
    )
    main(cfg)
