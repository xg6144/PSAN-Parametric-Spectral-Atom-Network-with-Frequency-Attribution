from dataclasses import dataclass, field, asdict
from typing import List

BASELINE = dict(
    atom_count=16,
    atom="gabor",
    init="morlet",
    isotropic=False,      # i.e. anisotropic envelope
    no_phase=False,       # i.e. phase is learnable
    atom_dropout=0.0,
    drop_path_rate=0.1,
    model_tag="psan_ti_M16_gabor_morlet",  # same as main-paper run
)

@dataclass
class AblationConfig:
    model_tag: str                        # unique tag; used for ckpt / eval
    axis: str                             # group label for the summary table
    description: str                      # human-readable one-liner

    atom_count: int = None
    atom: str = None                      # "gaussian" | "gabor"
    init: str = None                      # "morlet" | "random"
    isotropic: bool = None
    no_phase: bool = None
    atom_dropout: float = None

    def merged(self) -> dict:
        """Return a dict suitable for TrainConfig overrides, filling in
        None-valued fields with the baseline."""
        out = dict(BASELINE)
        out["model_tag"] = self.model_tag
        for k in (
            "atom_count", "atom", "init",
            "isotropic", "no_phase", "atom_dropout",
        ):
            v = getattr(self, k)
            if v is not None:
                out[k] = v
        return out

ABLATION_GRID: List[AblationConfig] = [
    AblationConfig(
        model_tag="psan_ti_M04_gabor_morlet",
        axis="atom_count",
        description="M = 4 (sparse atom bank)",
        atom_count=4,
    ),
    AblationConfig(
        model_tag="psan_ti_M08_gabor_morlet",
        axis="atom_count",
        description="M = 8",
        atom_count=8,
    ),
    AblationConfig(
        model_tag="psan_ti_M32_gabor_morlet",
        axis="atom_count",
        description="M = 32",
        atom_count=32,
    ),
    AblationConfig(
        model_tag="psan_ti_M64_gabor_morlet",
        axis="atom_count",
        description="M = 64 (dense atom bank)",
        atom_count=64,
    ),

    AblationConfig(
        model_tag="psan_ti_M16_gaussian_morlet",
        axis="atom_type",
        description="Gaussian atom (no carrier, learnable phase)",
        atom="gaussian",
    ),

    AblationConfig(
        model_tag="psan_ti_M16_gabor_random",
        axis="init",
        description="Random init (no Morlet dyadic prior)",
        init="random",
    ),

    AblationConfig(
        model_tag="psan_ti_M16_gabor_morlet_isotropic",
        axis="envelope",
        description="Isotropic Gaussian envelope (rho = 0)",
        isotropic=True,
    ),

    AblationConfig(
        model_tag="psan_ti_M16_gabor_morlet_nophase",
        axis="phase",
        description="Fixed phase (phi = 0)",
        no_phase=True,
    ),

    AblationConfig(
        model_tag="psan_ti_M16_gabor_morlet_ad0p1",
        axis="atom_dropout",
        description="Spectral atom dropout p = 0.1",
        atom_dropout=0.1,
    ),
    AblationConfig(
        model_tag="psan_ti_M16_gabor_morlet_ad0p2",
        axis="atom_dropout",
        description="Spectral atom dropout p = 0.2",
        atom_dropout=0.2,
    ),
]

BASELINE_SUMMARY_ENTRY = AblationConfig(
    model_tag=BASELINE["model_tag"],
    axis="baseline",
    description="Baseline (M=16, Gabor, Morlet, anisotropic, phase, no atom dropout)",
)

def all_configs_for_summary() -> List[AblationConfig]:
    """Baseline + 10 ablation runs, for the summary table."""
    return [BASELINE_SUMMARY_ENTRY] + ABLATION_GRID


if __name__ == "__main__":
    print(f"Baseline tag: {BASELINE['model_tag']}")
    print(f"Ablation runs: {len(ABLATION_GRID)}")
    print()
    print(f"{'axis':<16s} {'tag':<50s} description")
    print("-" * 110)
    for c in ABLATION_GRID:
        print(f"{c.axis:<16s} {c.model_tag:<50s} {c.description}")
