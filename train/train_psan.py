import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataclasses import dataclass
from typing import Optional
import torch
from models import psan as M
from train_base import BaseTrainConfig, setup_and_run

@dataclass
class TrainConfig(BaseTrainConfig):
    model_tag: str = "psan_ti"

    model: str = "psan_ti"
    drop_rate: float = 0.0
    drop_path_rate: float = 0.1

    atom_count: int = 16
    atom: str = "gabor"
    init: str = "morlet"
    isotropic: bool = False
    no_phase: bool = False
    atom_dropout: float = 0.2

    gfnet_pretrained: str = ""

NO_DECAY_KEYWORDS = (
    "norm",
    "mu", "log_sigma", "rho_raw", "theta", "log_amp", "phi",
    "w_real", "w_imag",
)

def build_model(cfg: TrainConfig, device):
    psan_kwargs = dict(
        M=cfg.atom_count,
        atom=cfg.atom,
        init=cfg.init,
        anisotropic=not cfg.isotropic,
        learn_phase=not cfg.no_phase,
        atom_dropout=cfg.atom_dropout,
    )
    factory = getattr(M, cfg.model)
    model = factory(
        num_classes=cfg.num_classes,
        psan_kwargs=psan_kwargs,
        drop_path_rate=cfg.drop_path_rate,
    )

    if cfg.gfnet_pretrained and os.path.exists(cfg.gfnet_pretrained):
        sd = torch.load(cfg.gfnet_pretrained, map_location="cpu")
        if "model" in sd:
            sd = sd["model"]
        if "state_dict" in sd:
            sd = sd["state_dict"]
        M.load_gfnet_pretrained(model, sd, verbose=True)

    pc = M.count_params(model)
    print(
        f"[PSAN] Params: total={pc['total']:,}, "
        f"filter={pc['filter_only']:,}"
    )
    return model.to(device)

def main(cfg: Optional[TrainConfig] = None):
    cfg = cfg or TrainConfig()
    setup_and_run(cfg, build_model, no_decay_keywords=NO_DECAY_KEYWORDS)

if __name__ == "__main__":
    cfg = TrainConfig(
        gpu_id=0,
        data_dir="/home/dongbeen/ML/Paper/PSAN/dataset_split",
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
        model_tag="psan_ti_M16_gabor_morlet_bw02",
        model="psan_ti",
        drop_rate=0.0,
        drop_path_rate=0.1,
        atom_count=16,
        atom="gabor",
        init="morlet",
        isotropic=False,
        no_phase=False,
        atom_dropout=0.2,
        gfnet_pretrained="",
    )
    main(cfg)
