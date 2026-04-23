import argparse
import gc
import os
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
TRAIN_DIR = PROJECT_ROOT / "train"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TRAIN_DIR))
sys.path.insert(0, str(HERE))

from ablation_configs import ABLATION_GRID, BASELINE, AblationConfig

def build_train_config(ab: AblationConfig, args: argparse.Namespace):
    """Construct the TrainConfig for one ablation run."""
    from train_psan import TrainConfig
    overrides = ab.merged()

    cfg = TrainConfig(
        gpu_id=args.gpu_id,
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        num_classes=args.num_classes,
        deterministic=True,
        early_stopping_patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        ema_decay=0.999,
        ema_warmup_epochs=20,

        model="psan_ti",
        drop_rate=0.0,
        drop_path_rate=overrides["drop_path_rate"],

        model_tag=overrides["model_tag"],
        atom_count=overrides["atom_count"],
        atom=overrides["atom"],
        init=overrides["init"],
        isotropic=overrides["isotropic"],
        no_phase=overrides["no_phase"],
        atom_dropout=overrides["atom_dropout"],

        psan_pretrained="",
    )
    return cfg


def checkpoint_exists(cfg) -> bool:
    """Return True if the best_*.pth for this run already exists."""
    cwd_best = Path.cwd() / f"best_{cfg.model_tag}.pth"
    return cwd_best.exists()


def run_one(ab: AblationConfig, args: argparse.Namespace) -> dict:
    """Launch one training run. Returns a result dict."""
    t0 = time.time()
    result = {
        "model_tag": ab.model_tag,
        "axis": ab.axis,
        "description": ab.description,
        "status": "pending",
        "elapsed_sec": None,
        "error": None,
    }

    cfg = build_train_config(ab, args)

    if args.skip_existing and checkpoint_exists(cfg):
        print(f"[SKIP] {ab.model_tag} — best_*.pth already exists.")
        result["status"] = "skipped_existing"
        result["elapsed_sec"] = 0.0
        return result

    if args.dry_run:
        print(f"[PLAN] {ab.axis:<16s} {ab.model_tag}")
        print(f"       atom={cfg.atom} M={cfg.atom_count} init={cfg.init} "
              f"iso={cfg.isotropic} no_phase={cfg.no_phase} "
              f"ad={cfg.atom_dropout}")
        result["status"] = "planned"
        return result

    print(f"\n{'='*78}\n[RUN] {ab.axis:<16s} {ab.model_tag}\n"
          f"      {ab.description}\n{'='*78}")
    try:
        from train_psan import main as train_main
        train_main(cfg)
        result["status"] = "ok"
    except Exception as e:
        result["status"] = "failed"
        result["error"] = f"{type(e).__name__}: {e}"
        print(f"[FAIL] {ab.model_tag} — {result['error']}")
        traceback.print_exc()
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        result["elapsed_sec"] = time.time() - t0

    return result

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--data-dir", type=str,
                    default="/home/dongbeen/ML/Paper/PSAN/dataset_split")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--num-classes", type=int, default=11)
    ap.add_argument("--epochs", type=int, default=300,
                    help="Maximum epochs; early stopping will end sooner.")
    ap.add_argument("--patience", type=int, default=20,
                    help="Early-stopping patience on val macro-F1.")
    ap.add_argument("--warmup-epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--only", type=str, nargs="+", default=None,
                    help="Run only configs whose model_tag contains one of "
                         "these substrings (e.g. M04 gaussian ad0p1).")
    ap.add_argument("--skip-axis", type=str, nargs="+", default=None,
                    help="Skip configs in these axes (atom_count | atom_type | "
                         "init | envelope | phase | atom_dropout).")
    ap.add_argument("--skip-existing", action="store_true", default=True,
                    help="Skip runs whose best_*.pth already exists (default: True).")
    ap.add_argument("--retrain-all", dest="skip_existing", action="store_false",
                    help="Retrain even if a checkpoint already exists.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan without launching any training.")

    args = ap.parse_args()

    grid = list(ABLATION_GRID)
    if args.only:
        grid = [c for c in grid if any(tok in c.model_tag for tok in args.only)]
    if args.skip_axis:
        grid = [c for c in grid if c.axis not in set(args.skip_axis)]
    if not grid:
        print("[INFO] After filtering, no configs remain. Nothing to do.")
        return

    print(f"[PLAN] {len(grid)} ablation run(s) queued:")
    for c in grid:
        print(f"   - {c.axis:<14s} {c.model_tag}")

    if args.dry_run:
        for c in grid:
            run_one(c, args)
        return

    results = []
    t_start = time.time()
    for i, c in enumerate(grid, 1):
        print(f"\n>>> [{i}/{len(grid)}]  {c.model_tag}")
        results.append(run_one(c, args))

    total_sec = time.time() - t_start
    ok = sum(r["status"] == "ok" for r in results)
    sk = sum(r["status"] == "skipped_existing" for r in results)
    fail = sum(r["status"] == "failed" for r in results)
    print("\n" + "=" * 78)
    print(f"Ablation grid finished in {total_sec/3600:.2f} h")
    print(f"  ok = {ok}   skipped = {sk}   failed = {fail}")
    print("=" * 78)
    for r in results:
        mark = {"ok": "[OK]", "skipped_existing": "[SK]",
                "failed": "[FAIL]", "planned": "[PL]"}.get(r["status"], "[??]")
        dur = "" if r["elapsed_sec"] is None else f"{r['elapsed_sec']/60:.1f}m"
        print(f"  {mark:<6s} {r['model_tag']:<55s} {dur}")
        if r["error"]:
            print(f"         error: {r['error']}")


if __name__ == "__main__":
    main()
