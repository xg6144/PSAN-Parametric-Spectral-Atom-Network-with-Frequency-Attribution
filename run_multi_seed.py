# run_multi_seed.py
"""
Usage
-----
    # run EVERY registered model with default 3 seeds
    python run_multi_seed.py --all
    python run_multi_seed.py                       # same thing (implicit --all)

    # run all, but skip two expensive ones
    python run_multi_seed.py --all --exclude swin_tiny resnet50

    # single model, custom seeds
    python run_multi_seed.py --model psan_best --seeds 42 123 2024

    # several models
    python run_multi_seed.py --model psan_best gfnet_tiny resnet50 \
                             --seeds 42 123 2024 777

    # preview what would be run — NO training happens
    python run_multi_seed.py --all --dry_run

    # eval-only (학습 스킵 — 기존 checkpoint가 이미 있다고 가정)
    python run_multi_seed.py --all --seeds 42 123 2024 --eval_only

Each model is registered below in MODEL_REGISTRY. To add a new model,
add an entry that plugs in (TrainConfig, train_main, EvalConfig, eval_main).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
for _p in (_REPO_ROOT,
           os.path.join(_REPO_ROOT, "train"),
           os.path.join(_REPO_ROOT, "eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
from train_base import find_latest_checkpoint_path

def _psan_recipe(*, atom_dropout: float = 0.2,
                 atom_count: int = 16,
                 atom: str = "gabor",
                 init: str = "morlet",
                 isotropic: bool = False,
                 no_phase: bool = False) -> Dict[str, Any]:
    from train_psan import TrainConfig as PSANTrainCfg, main as psan_train_main
    from eval_psan import EvalConfig as PSANEvalCfg, main as psan_eval_main

    def build_train_cfg(seed: int, tag: str, data_dir: str, gpu_id: int,
                        epochs: int) -> PSANTrainCfg:
        return PSANTrainCfg(
            gpu_id=gpu_id,
            data_dir=data_dir,
            img_size=224, batch_size=64,
            epochs=epochs, lr=1e-4, weight_decay=1e-3,
            num_workers=8, seed=seed, num_classes=11,
            deterministic=True, early_stopping_patience=20,
            warmup_epochs=15, ema_decay=0.999, ema_warmup_epochs=20,
            model_tag=tag, model="psan_ti",
            drop_rate=0.0, drop_path_rate=0.1,
            atom_count=atom_count, atom=atom, init=init,
            isotropic=isotropic, no_phase=no_phase,
            atom_dropout=atom_dropout, psan_pretrained="",
        )

    def build_eval_cfg(tag: str, ckpt: str, data_dir: str, gpu_id: int,
                       out_dir: str, seed: int) -> PSANEvalCfg:
        return PSANEvalCfg(
            gpu_id=gpu_id, data_dir=data_dir, split="test",
            img_size=224, batch_size=64, num_workers=8, seed=seed,
            num_classes=11,
            ckpt_path=ckpt, weight_source="auto",
            output_dir=out_dir, model_tag=tag,
            bootstrap=True, n_bootstrap=1000,
            measure_latency=True, latency_warmup=20,
            latency_iters=100, latency_batch_size=1,
        )

    return dict(
        build_train_cfg=build_train_cfg,
        train_main=psan_train_main,
        build_eval_cfg=build_eval_cfg,
        eval_main=psan_eval_main,
    )


def _timm_baseline_recipe(backbone_name: str, train_module: str,
                          eval_module: str) -> Dict[str, Any]:
    def _load():
        tm = __import__(train_module, fromlist=["TrainConfig", "main"])
        em = __import__(eval_module, fromlist=["EvalConfig", "main"])
        return tm, em

    def build_train_cfg(seed: int, tag: str, data_dir: str, gpu_id: int,
                        epochs: int):
        tm, _ = _load()
        return tm.TrainConfig(
            gpu_id=gpu_id, data_dir=data_dir,
            img_size=224, batch_size=64,
            epochs=epochs, lr=1e-4, weight_decay=1e-3,
            num_workers=8, seed=seed, num_classes=11,
            deterministic=True, early_stopping_patience=20,
            warmup_epochs=15, ema_decay=0.999, ema_warmup_epochs=20,
            backbone_name=backbone_name, model_tag=tag,
            pretrained=False, drop_rate=0.0, drop_path_rate=0.05,
        )

    def build_eval_cfg(tag: str, ckpt: str, data_dir: str, gpu_id: int,
                       out_dir: str, seed: int):
        _, em = _load()
        return em.EvalConfig(
            gpu_id=gpu_id, data_dir=data_dir, split="test",
            img_size=224, batch_size=64, num_workers=8, seed=seed,
            num_classes=11,
            ckpt_path=ckpt, weight_source="auto",
            output_dir=out_dir, model_tag=tag,
            bootstrap=True, n_bootstrap=1000,
            measure_latency=True, latency_warmup=20,
            latency_iters=100, latency_batch_size=1,
        )

    def train_main(cfg):
        tm, _ = _load()
        tm.main(cfg)

    def eval_main(cfg):
        _, em = _load()
        em.main(cfg)

    return dict(
        build_train_cfg=build_train_cfg,
        train_main=train_main,
        build_eval_cfg=build_eval_cfg,
        eval_main=eval_main,
    )

def _build_registry() -> Dict[str, Dict[str, Any]]:
    reg: Dict[str, Dict[str, Any]] = {}

    reg["psan_best"] = _psan_recipe(atom_dropout=0.2)
    reg["psan_base"] = _psan_recipe(atom_dropout=0.0)
    reg["psan_gaussian"] = _psan_recipe(atom_dropout=0.0, atom="gaussian")
    reg["psan_random"] = _psan_recipe(atom_dropout=0.0, init="random")

    reg["resnet50"] = _timm_baseline_recipe("resnet50", "train_resnet", "eval_resnet")
    reg["efficientnet_b0"] = _timm_baseline_recipe(
        "efficientnet_b0", "train_efficientnet", "eval_efficientnet"
    )
    reg["deit_tiny"] = _timm_baseline_recipe(
        "deit_tiny_patch16_224", "train_deit", "eval_deit"
    )
    reg["swin_tiny"] = _timm_baseline_recipe(
        "swin_tiny_patch4_window7_224", "train_swintransformer", "eval_swintransformer"
    )
    reg["convnextv2_atto"] = _timm_baseline_recipe(
        "convnextv2_atto", "train_convnextv2", "eval_convnextv2"
    )
    reg["gfnet_tiny"] = _timm_baseline_recipe(
        "gfnet_tiny", "train_gfnet", "eval_gfnet"
    )
    reg["afno_tiny"] = _timm_baseline_recipe(
        "afno_tiny", "train_afno", "eval_afno"
    )
    reg["mlp_mixer_tiny"] = _timm_baseline_recipe(
        "mlp_mixer_tiny", "train_mlp_mixer", "eval_mlp_mixer"
    )

    return reg


MODEL_REGISTRY = _build_registry()

def _ckpt_path(tag: str) -> Optional[str]:
    """Return the newest checkpoint for a tag from train_results or legacy paths."""
    ckpt_path = find_latest_checkpoint_path(tag, prefix="best")
    return str(ckpt_path) if ckpt_path is not None else None

def _tag_for(model_key: str, seed: int, base_tag: Optional[str]) -> str:
    """Checkpoint / output naming: `{base}_seed{seed}`."""
    base = base_tag or model_key
    return f"{base}_seed{seed}"

def _extract_metrics(eval_json_path: str) -> Dict[str, float]:
    """Pull headline numbers out of the eval JSON produced by eval_*.py."""
    with open(eval_json_path) as f:
        data = json.load(f)
    m = data.get("metrics", {})
    out: Dict[str, float] = {}
    for k in ("accuracy", "balanced_accuracy",
              "macro_f1", "weighted_f1",
              "macro_precision", "macro_recall",
              "cohen_kappa", "mcc",
              "macro_auc_ovr", "weighted_auc_ovr", "macro_auc_ovo"):
        if k in m:
            out[k] = float(m[k])
    mi = data.get("model_info", {})
    for k in ("total_params", "total_params_M",
              "flops", "flops_G", "macs_G"):
        if k in mi and mi[k] is not None:
            out[k] = float(mi[k])
    lat = m.get("latency", {}) or {}
    for k in ("mean_ms", "p50_ms", "p95_ms", "throughput_imgs_per_sec"):
        if k in lat and lat[k] is not None:
            out[f"lat_{k}"] = float(lat[k])
    return out


def _run_single(model_key: str, seed: int, *,
                base_tag: Optional[str],
                data_dir: str, gpu_id: int, epochs: int,
                output_dir: str, eval_only: bool,
                force_retrain: bool) -> Dict[str, Any]:
    """Run one (model, seed) combination. Returns metric dict + metadata."""
    recipe = MODEL_REGISTRY[model_key]
    tag = _tag_for(model_key, seed, base_tag)
    ckpt = _ckpt_path(tag)

    if not eval_only:
        if ckpt is not None and not force_retrain:
            print(f"[skip-train] {ckpt} already exists — reusing.")
        else:
            print(f"\n{'='*72}\n[TRAIN] model={model_key} seed={seed} tag={tag}\n{'='*72}")
            train_cfg = recipe["build_train_cfg"](
                seed=seed, tag=tag, data_dir=data_dir,
                gpu_id=gpu_id, epochs=epochs,
            )
            t0 = time.time()
            recipe["train_main"](train_cfg)
            print(f"[train-done] {tag} — elapsed={(time.time()-t0)/60:.1f} min")
            ckpt = _ckpt_path(tag)

    if ckpt is None or not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"Expected checkpoint for tag={tag} not found — training did not "
            f"produce a best_*.pth under train_results/. Check the training log."
        )

    print(f"\n{'-'*72}\n[EVAL] model={model_key} seed={seed} tag={tag}\n{'-'*72}")
    eval_cfg = recipe["build_eval_cfg"](
        tag=tag, ckpt=ckpt, data_dir=data_dir, gpu_id=gpu_id,
        out_dir=output_dir, seed=seed,
    )
    os.makedirs(output_dir, exist_ok=True)
    recipe["eval_main"](eval_cfg)

    json_path = os.path.join(output_dir, f"{tag}_eval_metrics.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Expected eval metrics at {json_path} but not found. "
            f"Check that the eval script writes into output_dir={output_dir}."
        )
    metrics = _extract_metrics(json_path)
    metrics.update({"model": model_key, "seed": seed, "tag": tag,
                    "ckpt": ckpt, "eval_json": json_path})
    return metrics

METRIC_COLS = [
    "accuracy", "balanced_accuracy",
    "macro_f1", "weighted_f1",
    "macro_precision", "macro_recall",
    "cohen_kappa", "mcc",
    "macro_auc_ovr", "weighted_auc_ovr", "macro_auc_ovo",
    "lat_mean_ms", "lat_throughput_imgs_per_sec",
]


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    from math import sqrt
    Z = 1.959963984540054  # two-sided 95% normal

    rows: List[Dict[str, Any]] = []
    for model, sub in df.groupby("model"):
        n = len(sub)
        row: Dict[str, Any] = {"model": model, "n_seeds": n,
                               "seeds": sorted(sub["seed"].tolist())}
        for col in METRIC_COLS:
            if col not in sub.columns:
                continue
            vals = sub[col].dropna().values.astype(float)
            if vals.size == 0:
                continue
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
            sem = std / sqrt(vals.size) if vals.size > 1 else 0.0
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
            row[f"{col}_ci95_lo"] = mean - Z * sem
            row[f"{col}_ci95_hi"] = mean + Z * sem
        rows.append(row)
    agg = pd.DataFrame(rows)
  
    if "macro_f1_mean" in agg.columns:
        agg = agg.sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)
    return agg

def _pretty_print(agg: pd.DataFrame) -> None:
    if agg.empty:
        print("[aggregate] empty — nothing to report.")
        return
    print("\n" + "=" * 96)
    print(" AGGREGATED RESULTS  (mean ± std across seeds)")
    print("=" * 96)
    header = f"{'Model':<28s} | {'n':>2s} | {'Accuracy':>16s} | {'Macro F1':>16s} | {'Macro AUC':>16s} | {'Kappa':>16s}"
    print(header)
    print("-" * len(header))

    def _fmt(m, s):
        if s == 0.0:
            return f"{m:.4f}"
        return f"{m:.4f} ± {s:.4f}"

    for _, r in agg.iterrows():
        print(
            f"{r['model']:<28s} | {int(r['n_seeds']):>2d} | "
            f"{_fmt(r.get('accuracy_mean', float('nan')), r.get('accuracy_std', 0.0)):>16s} | "
            f"{_fmt(r.get('macro_f1_mean', float('nan')), r.get('macro_f1_std', 0.0)):>16s} | "
            f"{_fmt(r.get('macro_auc_ovr_mean', float('nan')), r.get('macro_auc_ovr_std', 0.0)):>16s} | "
            f"{_fmt(r.get('cohen_kappa_mean', float('nan')), r.get('cohen_kappa_std', 0.0)):>16s}"
        )
    print("=" * 96)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-seed training + evaluation runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model", nargs="+", default=None,
        choices=sorted(MODEL_REGISTRY.keys()),
        help="Models to run (space-separated). If omitted, ALL models in "
             "the registry are run in their declared order. Use --exclude "
             "to skip some.",
    )
    p.add_argument(
        "--all", action="store_true",
        help="Explicit flag to run every registered model. Equivalent to "
             "omitting --model.",
    )
    p.add_argument(
        "--exclude", nargs="+", default=[],
        choices=sorted(MODEL_REGISTRY.keys()),
        help="Model keys to skip when running --all.",
    )
    p.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 123, 2024],
        help="Seeds to use.",
    )
    p.add_argument(
        "--base_tag", type=str, default=None,
        help="Override the base tag (defaults to model key). "
             "Seed suffix is always appended: `{base_tag}_seed{seed}`.",
    )
    p.add_argument(
        "--data_dir", type=str,
        default="/home/dongbeen/ML/Paper/AnchorGViT/dataset_split",
        help="Root of the dataset split folder.",
    )
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument(
        "--epochs", type=int, default=300,
        help="Max epochs (early stopping usually halts much earlier).",
    )
    p.add_argument(
        "--output_dir", type=str, default="multi_seed_results",
        help="Where eval JSON / aggregate CSV go.",
    )
    p.add_argument(
        "--eval_only", action="store_true",
        help="Skip training; expect a saved checkpoint already exists.",
    )
    p.add_argument(
        "--force_retrain", action="store_true",
        help="Retrain even if a saved checkpoint already exists.",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Print the plan (models × seeds, and which checkpoints are "
             "already on disk) and exit WITHOUT training or evaluating.",
    )
    return p.parse_args()


def _resolve_models(args: argparse.Namespace) -> List[str]:
    """Turn CLI args into an ordered list of model keys to run."""
    if args.model is None or args.all:
        selected = list(MODEL_REGISTRY.keys())
    else:
        selected = list(args.model)

    if args.exclude:
        excluded = set(args.exclude)
        selected = [m for m in selected if m not in excluded]

    seen: set = set()
    ordered: List[str] = []
    for m in selected:
        if m not in seen:
            seen.add(m)
            ordered.append(m)
    return ordered


def _print_plan(models: List[str], seeds: List[int], output_dir: str,
                eval_only: bool, force_retrain: bool) -> None:
    total = len(models) * len(seeds)
    print("\n" + "=" * 76)
    print(f" PLAN: {len(models)} models × {len(seeds)} seeds = {total} runs")
    print("=" * 76)
    print(f"{'idx':>3s}  {'model':<28s}  {'seed':>6s}  {'tag':<42s}  status")
    print("-" * 100)
    idx = 0
    skipped_train = 0
    skipped_eval = 0
    to_train = 0
    for m in models:
        for s in seeds:
            idx += 1
            tag = f"{m}_seed{s}"
            ckpt = _ckpt_path(tag)
            eval_json = os.path.join(output_dir, f"{tag}_eval_metrics.json")
            if ckpt is not None:
                if os.path.exists(eval_json) and not force_retrain:
                    status = "✓ ckpt + eval exist"
                    skipped_train += 1
                    skipped_eval += 1
                elif force_retrain:
                    status = "RETRAIN (forced)"
                    to_train += 1
                else:
                    status = "ckpt exists — will EVAL only"
                    skipped_train += 1
            else:
                if eval_only:
                    status = "✗ MISSING ckpt (eval_only fails)"
                else:
                    status = "TRAIN + EVAL"
                    to_train += 1
            print(f"{idx:>3d}  {m:<28s}  {s:>6d}  {tag:<42s}  {status}")
    print("-" * 100)
    print(f"  → to train: {to_train},  train-skipped: {skipped_train},  "
          f"eval-skipped: {skipped_eval}")
    print("=" * 76)


def main() -> None:
    args = parse_args()

    models = _resolve_models(args)
    if not models:
        print("[runner] no models to run (empty selection after --exclude).")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    _print_plan(
        models, args.seeds, args.output_dir,
        eval_only=args.eval_only, force_retrain=args.force_retrain,
    )
    if args.dry_run:
        print("\n[dry-run] exiting without training.")
        return

    total_runs = len(models) * len(args.seeds)
    per_run_records: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    print(f"\n[runner] models={models}")
    print(f"[runner] seeds={args.seeds}")
    print(f"[runner] data_dir={args.data_dir} gpu={args.gpu_id} "
          f"output_dir={args.output_dir}")

    run_idx = 0
    run_start = time.time()
    for model_key in models:
        if model_key not in MODEL_REGISTRY:
            print(f"[runner] unknown model '{model_key}', skipping.")
            continue
        for seed in args.seeds:
            run_idx += 1
            elapsed = time.time() - run_start
            eta = (elapsed / max(run_idx - 1, 1)) * (total_runs - run_idx + 1) \
                if run_idx > 1 else float("nan")
            print(
                f"\n>>> [{run_idx}/{total_runs}] model={model_key} seed={seed}  "
                f"elapsed={elapsed/60:.1f} min"
                + (f"  ETA≈{eta/60:.1f} min" if run_idx > 1 else "")
            )
            try:
                rec = _run_single(
                    model_key, seed,
                    base_tag=args.base_tag,
                    data_dir=args.data_dir,
                    gpu_id=args.gpu_id,
                    epochs=args.epochs,
                    output_dir=args.output_dir,
                    eval_only=args.eval_only,
                    force_retrain=args.force_retrain,
                )
                per_run_records.append(rec)
            except Exception as e:
                print(f"[runner][error] model={model_key} seed={seed} → {e}")
                errors.append({"model": model_key, "seed": seed, "error": repr(e)})

            if per_run_records:
                per_run_csv = os.path.join(args.output_dir, "per_run_metrics.csv")
                pd.DataFrame(per_run_records).to_csv(per_run_csv, index=False)

    total_elapsed = (time.time() - run_start) / 60.0
    print(f"\n[runner] all runs finished — total elapsed = {total_elapsed:.1f} min")

    if per_run_records:
        df = pd.DataFrame(per_run_records)
        per_run_csv = os.path.join(args.output_dir, "per_run_metrics.csv")
        df.to_csv(per_run_csv, index=False)
        print(f"[saved] {per_run_csv}  (n={len(df)} runs)")
    else:
        df = pd.DataFrame()
        print("[runner] no runs completed successfully.")

    if not df.empty:
        agg = _aggregate(df)
        agg_csv = os.path.join(args.output_dir, "aggregate_metrics.csv")
        agg.to_csv(agg_csv, index=False)
        print(f"[saved] {agg_csv}")

        _save_latex_table(
            agg, os.path.join(args.output_dir, "aggregate_metrics.tex"),
        )

        _pretty_print(agg)

    manifest = {
        "args": vars(args),
        "resolved_models": models,
        "registry_keys": sorted(MODEL_REGISTRY.keys()),
        "total_elapsed_min": total_elapsed,
        "per_run_records": per_run_records,
        "errors": errors,
    }
    manifest_path = os.path.join(args.output_dir, "run_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[saved] {manifest_path}")
    if errors:
        print(f"[runner] {len(errors)} failures. See run_manifest.json.")

def _save_latex_table(agg: pd.DataFrame, path: str) -> None:
    """Compact booktabs table for the paper (Macro F1 ± std, Acc ± std, AUC ± std)."""
    if agg.empty:
        return
    lines = [
        "% Generated by run_multi_seed.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Multi-seed test performance (mean $\pm$ std over $n$ seeds).}",
        r"\label{tab:multi_seed}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & $n$ & Accuracy & Macro F1 & Macro AUC-OvR \\",
        r"\midrule",
    ]
    for _, r in agg.iterrows():
        def _cell(metric: str) -> str:
            m, s = r.get(f"{metric}_mean", float("nan")), r.get(f"{metric}_std", 0.0)
            if pd.isna(m):
                return "--"
            if s == 0:
                return f"{m:.4f}"
            return f"${m:.4f} \\pm {s:.4f}$"
        lines.append(
            f"{r['model'].replace('_', r'\_')} & {int(r['n_seeds'])} & "
            f"{_cell('accuracy')} & {_cell('macro_f1')} & {_cell('macro_auc_ovr')} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[saved] {path}")

if __name__ == "__main__":
    main()
