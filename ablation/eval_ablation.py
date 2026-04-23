import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
EVAL_DIR = PROJECT_ROOT / "eval"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(HERE))

from ablation_configs import all_configs_for_summary

def find_checkpoint(model_tag: str, search_dirs) -> Path:
    """Return the first existing best_<tag>.pth across search_dirs."""
    for d in search_dirs:
        p = Path(d) / f"best_{model_tag}.pth"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No best_{model_tag}.pth found in any of: "
        + ", ".join(str(d) for d in search_dirs)
    )

def run_one_eval(ab, ckpt_path: Path, args: argparse.Namespace) -> dict:
    """Run eval_psan.main(cfg) for a single checkpoint."""
    from eval_psan import EvalConfig, main as eval_main  # noqa: WPS433
    cfg = EvalConfig(
        gpu_id=args.gpu_id,
        data_dir=args.data_dir,
        split=args.split,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        num_classes=args.num_classes,
        ckpt_path=str(ckpt_path),
        weight_source="auto",
        output_dir=args.output_dir,
        model_tag=ab.model_tag,
        bootstrap=True,
        n_bootstrap=args.n_bootstrap,
        measure_latency=args.measure_latency,
        latency_warmup=args.latency_warmup,
        latency_iters=args.latency_iters,
        latency_batch_size=args.latency_batch_size,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    eval_main(cfg)

def _flatten_row(ab, metrics_json: dict) -> dict:
    m = metrics_json["metrics"]
    mi = metrics_json.get("model_info", {})
    b = m.get("bootstrap_ci", {})
    lat = m.get("latency", {})
    return dict(
        axis=ab.axis,
        model_tag=ab.model_tag,
        description=ab.description,
        atom_count=mi.get("atom_count"),
        atom=mi.get("atom"),
        init=mi.get("init"),
        isotropic=mi.get("isotropic"),
        no_phase=mi.get("no_phase"),
        atom_dropout=mi.get("atom_dropout_train"),
        total_params=mi.get("total_params"),
        total_params_M=mi.get("total_params_M"),
        filter_only_params=mi.get("filter_only_params"),
        flops_G=mi.get("flops_G"),
        latency_mean_ms=lat.get("mean_ms"),
        # metrics
        accuracy=m.get("accuracy"),
        macro_f1=m.get("macro_f1"),
        weighted_f1=m.get("weighted_f1"),
        macro_auc=m.get("macro_auc_ovr"),
        cohen_kappa=m.get("cohen_kappa"),
        balanced_accuracy=m.get("balanced_accuracy"),
        # CIs
        macro_f1_ci_lo=b.get("macro_f1", {}).get("ci95_lower"),
        macro_f1_ci_hi=b.get("macro_f1", {}).get("ci95_upper"),
        accuracy_ci_lo=b.get("accuracy", {}).get("ci95_lower"),
        accuracy_ci_hi=b.get("accuracy", {}).get("ci95_upper"),
    )


def collect_summary(output_dir: Path) -> pd.DataFrame:
    """Aggregate every <tag>_eval_metrics.json into a dataframe."""
    rows = []
    for ab in all_configs_for_summary():
        jpath = output_dir / f"{ab.model_tag}_eval_metrics.json"
        if not jpath.exists():
            print(f"[WARN] metrics JSON missing for {ab.model_tag}: {jpath}")
            continue
        with open(jpath, "r") as f:
            md = json.load(f)
        rows.append(_flatten_row(ab, md))
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--data-dir", type=str,
                    default="/home/dongbeen/ML/Paper/AnchorGViT/dataset_split")
    ap.add_argument("--split", type=str, default="test",
                    choices=["val", "test"])
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--num-classes", type=int, default=11)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--output-dir", type=str, default="ablation_results")
    ap.add_argument("--ckpt-dirs", type=str, nargs="+", default=None,
                    help="Directories to search for best_<tag>.pth "
                         "(default: CWD + ablation/ + checkpoints/).")

    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--measure-latency", action="store_true", default=True)
    ap.add_argument("--no-latency", dest="measure_latency", action="store_false")
    ap.add_argument("--latency-warmup", type=int, default=20)
    ap.add_argument("--latency-iters", type=int, default=100)
    ap.add_argument("--latency-batch-size", type=int, default=1)

    ap.add_argument("--only", type=str, nargs="+", default=None,
                    help="Evaluate only tags containing one of these substrings.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip runs whose eval_metrics.json already exists.")
    ap.add_argument("--summary-only", action="store_true",
                    help="Skip the evaluation loop; just build the summary CSV.")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    search_dirs = args.ckpt_dirs or [
        Path.cwd(),
        HERE,
        PROJECT_ROOT,
        PROJECT_ROOT / "checkpoints",
        PROJECT_ROOT / "ablation" / "checkpoints",
    ]

    configs = all_configs_for_summary()
    if args.only:
        configs = [c for c in configs
                   if any(tok in c.model_tag for tok in args.only)]

    if not args.summary_only:
        for i, ab in enumerate(configs, 1):
            jpath = output_dir / f"{ab.model_tag}_eval_metrics.json"
            if args.skip_existing and jpath.exists():
                print(f"[SKIP] {ab.model_tag} — eval JSON already exists.")
                continue
            try:
                ckpt = find_checkpoint(ab.model_tag, search_dirs)
            except FileNotFoundError as e:
                print(f"[WARN] {e}")
                continue
            t0 = time.time()
            print(f"\n>>> [{i}/{len(configs)}] eval: {ab.model_tag}")
            run_one_eval(ab, ckpt, args)
            print(f"    done in {time.time() - t0:.1f}s")

    df = collect_summary(output_dir)
    if df.empty:
        print("[WARN] No evaluation results aggregated.")
        return

    axis_order = {"baseline": 0, "atom_count": 1, "atom_type": 2,
                  "init": 3, "envelope": 4, "phase": 5, "atom_dropout": 6}
    df["_axis_ord"] = df["axis"].map(axis_order).fillna(99).astype(int)
    df = df.sort_values(["_axis_ord", "macro_f1"],
                        ascending=[True, False]).drop(columns=["_axis_ord"])

    csv_path = output_dir / "ablation_summary.csv"
    json_path = output_dir / "ablation_summary.json"
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)
    print(f"\n[SAVE] {csv_path}")
    print(f"[SAVE] {json_path}")
    print()

    cols_to_show = [
        "axis", "model_tag", "total_params_M",
        "accuracy", "macro_f1", "macro_f1_ci_lo", "macro_f1_ci_hi",
        "macro_auc",
    ]
    print(df[cols_to_show].to_string(index=False,
                                     float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
