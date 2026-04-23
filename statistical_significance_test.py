"""
Statistical Significance Test — PSAN vs Baselines & Ablations
=============================================================
이 스크립트는 IEEE Access 논문 투고를 위해 모델 간 성능 차이의
통계적 유의성을 검증합니다.

수행하는 검정:
  1. McNemar's test   — 두 모델의 sample-level 정오분류 패턴 비교
  2. Paired Bootstrap — Macro F1 / Balanced Acc / AUC / Kappa / MCC 등
                        metric-level 차이의 95% CI와 p-value 산출
  3. Bonferroni 보정   — 다중 비교 보정 적용

입력:
  eval_results/ 디렉토리 아래 각 모델의 *_predictions.csv
  (columns: path, true_label, true_class, pred_label, pred_class, correct,
   prob_<class_name> ...)

출력:
  - significance_mcnemar.csv        : McNemar 검정 결과
  - significance_paired_bootstrap.csv: Paired Bootstrap 검정 결과
  - significance_summary.csv         : 논문 삽입용 요약 테이블
  - significance_summary.tex         : LaTeX 테이블 (IEEE 포맷)
  - significance_heatmap.png         : p-value 히트맵 시각화

Usage:
  python statistical_significance_test.py \
      --eval_dir eval_results \
      --proposed psan_tiny_bw02 \
      --n_bootstrap 10000 \
      --seed 42 \
      --output_dir significance_results
"""

import argparse
import os
import json
import warnings
from itertools import combinations
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)


# ======================================================================
#  Configuration
# ======================================================================

# ---- 베이스라인 모델 (eval 시 사용된 model_tag) ----
BASELINE_MODELS = {
    "resnet50":       "ResNet-50",
    "swin_tiny":      "Swin-Tiny",
    "efficientnet_b0":"EfficientNet-B0",
    "convnextv2_atto":"ConvNeXtV2-Atto",
    "deit_tiny":      "DeiT-Tiny",
    "gfnet_tiny":     "GFNet-Tiny",
    "mlp_mixer_tiny": "MLP-Mixer-Tiny",
    "afno_tiny":      "AFNO-Tiny",
}

# ---- Ablation 모델 ----
ABLATION_MODELS = {
    "psan_m4":              "PSAN M=4",
    "psan_m8":              "PSAN M=8",
    "psan_m16":             "PSAN M=16",
    "psan_m32":             "PSAN M=32",
    "psan_m64":             "PSAN M=64",
    "psan_gaussian_morlet": "Gaussian+Morlet",
    "psan_gabor_random":    "Gabor+Random",
    "psan_no_phase":        "No Phase",
    "psan_isotropic":       "Isotropic",
    "psan_bw01":            "BW=0.1",
}

# ---- 제안 모델 ----
PROPOSED_TAG = "pspn_gfnet_ti_M16_gabor_morlet_bd0p2"
PROPOSED_NAME = "PSAN-Tiny (BW=0.2)"

ALL_MODELS = {**BASELINE_MODELS, **ABLATION_MODELS, PROPOSED_TAG: PROPOSED_NAME}


# ======================================================================
#  1. 데이터 로딩
# ======================================================================

def discover_prediction_csvs(eval_dir: str) -> Dict[str, str]:
    """eval_dir 안의 *_predictions.csv 파일을 검색하여 {model_tag: filepath} 반환."""
    found = {}
    if not os.path.isdir(eval_dir):
        raise FileNotFoundError(f"eval_dir not found: {eval_dir}")

    for fname in sorted(os.listdir(eval_dir)):
        if fname.endswith("_predictions.csv"):
            tag = fname.replace("_predictions.csv", "")
            found[tag] = os.path.join(eval_dir, fname)
    return found


def load_predictions(csv_path: str) -> pd.DataFrame:
    """prediction CSV를 로드하여 정렬된 DataFrame 반환."""
    df = pd.read_csv(csv_path)
    required = {"path", "true_label", "pred_label", "correct"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns in {csv_path}: {required - set(df.columns)}")
    # path 기준 정렬 (모든 모델의 샘플 순서를 동일하게)
    df = df.sort_values("path").reset_index(drop=True)
    return df


def extract_probs(df: pd.DataFrame) -> Optional[np.ndarray]:
    """prob_* 컬럼이 있으면 numpy array로 반환."""
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if not prob_cols:
        return None
    return df[prob_cols].values


# ======================================================================
#  2. McNemar's Test
# ======================================================================

def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """
    McNemar's exact test (scipy) + continuity-corrected chi-squared.

    contingency table:
                 Model B correct   Model B wrong
    Model A correct      n00            n01
    Model A wrong        n10            n11

    비교 대상: n01 vs n10 (discordant pairs)
    """
    assert len(correct_a) == len(correct_b)
    n = len(correct_a)

    # contingency counts
    both_correct = int(np.sum(correct_a & correct_b))
    a_only       = int(np.sum(correct_a & ~correct_b))   # A맞고 B틀림
    b_only       = int(np.sum(~correct_a & correct_b))   # B맞고 A틀림
    both_wrong   = int(np.sum(~correct_a & ~correct_b))

    discordant = a_only + b_only

    # ---- exact binomial (McNemar's exact) ----
    from scipy.stats import binomtest
    if discordant == 0:
        p_exact = 1.0
    else:
        try:
            result = binomtest(a_only, discordant, 0.5, alternative='two-sided')
            p_exact = result.pvalue
        except AttributeError:
            # scipy < 1.7 fallback
            p_exact = binom_test(a_only, discordant, 0.5)

    # ---- chi-squared with continuity correction ----
    if discordant == 0:
        chi2 = 0.0
        p_chi2 = 1.0
    else:
        chi2 = (abs(a_only - b_only) - 1) ** 2 / discordant
        from scipy.stats import chi2 as chi2_dist
        p_chi2 = 1 - chi2_dist.cdf(chi2, df=1)

    return {
        "n_samples": n,
        "both_correct": both_correct,
        "a_correct_b_wrong": a_only,
        "b_correct_a_wrong": b_only,
        "both_wrong": both_wrong,
        "discordant_pairs": discordant,
        "chi2_cc": round(chi2, 4),
        "p_value_chi2": p_chi2,
        "p_value_exact": p_exact,
    }


# ======================================================================
#  3. Paired Bootstrap Test
# ======================================================================

def compute_metrics(y_true, y_pred, probs=None, num_classes=11):
    """단일 metric 딕셔너리 반환."""
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if probs is not None:
        try:
            y_bin = label_binarize(y_true, classes=list(range(num_classes)))
            results["macro_auc"] = roc_auc_score(
                y_bin, probs, average="macro", multi_class="ovr"
            )
        except Exception:
            results["macro_auc"] = np.nan
    return results


def paired_bootstrap_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    probs_a: Optional[np.ndarray],
    probs_b: Optional[np.ndarray],
    n_bootstrap: int = 10000,
    seed: int = 42,
    num_classes: int = 11,
    metric_keys: Optional[List[str]] = None,
) -> Dict[str, dict]:
    """
    Paired bootstrap test로 두 모델의 metric 차이 (A - B)의
    95% CI와 p-value를 계산.

    p-value = 2 × min(P(Δ ≤ 0), P(Δ ≥ 0))  (two-sided)

    Returns:
        {metric_name: {"observed_diff", "mean_diff", "ci95_lower", "ci95_upper",
                        "p_value", "a_better_pct"}}
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    if metric_keys is None:
        metric_keys = [
            "accuracy", "balanced_accuracy", "macro_f1",
            "weighted_f1", "cohen_kappa", "mcc",
        ]
        if probs_a is not None and probs_b is not None:
            metric_keys.append("macro_auc")

    # observed
    obs_a = compute_metrics(y_true, pred_a, probs_a, num_classes)
    obs_b = compute_metrics(y_true, pred_b, probs_b, num_classes)

    # bootstrap
    diffs = {k: [] for k in metric_keys}

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        pa = pred_a[idx]
        pb = pred_b[idx]
        pra = probs_a[idx] if probs_a is not None else None
        prb = probs_b[idx] if probs_b is not None else None

        ma = compute_metrics(yt, pa, pra, num_classes)
        mb = compute_metrics(yt, pb, prb, num_classes)

        for k in metric_keys:
            if k in ma and k in mb:
                diffs[k].append(ma[k] - mb[k])

    results = {}
    for k in metric_keys:
        d = np.array(diffs[k])
        observed = obs_a.get(k, 0) - obs_b.get(k, 0)

        # two-sided p-value
        p_lower = np.mean(d <= 0)  # A가 B 이하인 비율
        p_upper = np.mean(d >= 0)  # A가 B 이상인 비율
        p_value = 2 * min(p_lower, p_upper)
        p_value = min(p_value, 1.0)

        results[k] = {
            "model_a_value": obs_a.get(k, np.nan),
            "model_b_value": obs_b.get(k, np.nan),
            "observed_diff": round(observed, 6),
            "mean_diff": round(np.mean(d), 6),
            "std_diff": round(np.std(d), 6),
            "ci95_lower": round(np.percentile(d, 2.5), 6),
            "ci95_upper": round(np.percentile(d, 97.5), 6),
            "p_value": round(p_value, 6),
            "a_better_pct": round(np.mean(d > 0) * 100, 2),
        }

    return results


# ======================================================================
#  4. 유의성 수준 표기
# ======================================================================

def significance_stars(p: float, bonferroni_n: int = 1) -> str:
    """Bonferroni 보정 후 유의성 기호 반환."""
    p_adj = min(p * bonferroni_n, 1.0)
    if p_adj < 0.001:
        return "***"
    elif p_adj < 0.01:
        return "**"
    elif p_adj < 0.05:
        return "*"
    else:
        return "n.s."


# ======================================================================
#  5. 시각화
# ======================================================================

def plot_pvalue_heatmap(
    mcnemar_df: pd.DataFrame,
    output_path: str,
    proposed_name: str,
    n_comparisons: int,
):
    """McNemar p-value 히트맵 + Bonferroni 보정선 표시."""
    pivot = mcnemar_df.pivot_table(
        index="Model A", columns="Model B", values="p_exact", aggfunc="first"
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    log_p = -np.log10(pivot.values.astype(float) + 1e-300)

    sns.heatmap(
        log_p,
        xticklabels=pivot.columns,
        yticklabels=pivot.index,
        annot=pivot.applymap(lambda x: f"{x:.1e}" if x < 0.05 else f"{x:.3f}").values,
        fmt="",
        cmap="YlOrRd",
        cbar_kws={"label": r"$-\log_{10}(p)$"},
        ax=ax,
        linewidths=0.5,
    )

    # Bonferroni threshold lines
    bonf_thresh = 0.05 / n_comparisons
    ax.set_title(
        f"McNemar's Test — Pairwise p-values\n"
        f"(Bonferroni threshold: p < {bonf_thresh:.1e} for α=0.05, "
        f"n_comparisons={n_comparisons})",
        fontsize=11,
    )
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {output_path}")


def plot_bootstrap_ci(
    bs_results: List[dict],
    output_path: str,
    proposed_name: str,
    metric: str = "macro_f1",
):
    """Bootstrap CI 차이를 forest plot으로 시각화."""
    rows = [r for r in bs_results if metric in r["metrics"]]
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(rows) * 0.6)))

    labels = []
    diffs = []
    lows = []
    highs = []
    colors = []

    for r in rows:
        m = r["metrics"][metric]
        label = f"{r['model_b_name']}"
        labels.append(label)
        diffs.append(m["observed_diff"])
        lows.append(m["ci95_lower"])
        highs.append(m["ci95_upper"])
        # 유의하면 빨간색, 아니면 회색
        sig = m["p_value"] < 0.05 / len(rows)
        colors.append("#d32f2f" if sig else "#9e9e9e")

    y_pos = np.arange(len(labels))

    ax.barh(y_pos, diffs, xerr=[
        [d - l for d, l in zip(diffs, lows)],
        [h - d for d, h in zip(diffs, highs)],
    ], align="center", color=colors, alpha=0.8, height=0.5,
        error_kw=dict(capsize=3, linewidth=1.2))

    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(f"Δ {metric} ({proposed_name} − Baseline)", fontsize=10)
    ax.set_title(
        f"Paired Bootstrap — {metric} Difference\n"
        f"(Red = significant after Bonferroni correction at α=0.05)",
        fontsize=11,
    )
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {output_path}")


# ======================================================================
#  6. LaTeX 테이블 생성
# ======================================================================

def generate_latex_table(summary_df: pd.DataFrame, output_path: str):
    """IEEE Access 포맷의 LaTeX 테이블 생성."""
    lines = []
    lines.append(r"\begin{table}[!t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Statistical Significance of PSAN-Tiny (BW=0.2) vs.\ Baseline Models}")
    lines.append(r"\label{tab:significance}")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{l c c c c c}")
    lines.append(r"\hline\hline")
    lines.append(
        r"\textbf{Comparison} & \textbf{$\Delta$F1} & "
        r"\textbf{95\% CI} & \textbf{$p$-value} & "
        r"\textbf{McNemar $p$} & \textbf{Sig.} \\"
    )
    lines.append(r"\hline")

    for _, row in summary_df.iterrows():
        model_b = row["Model B"]
        delta_f1 = row.get("delta_macro_f1", 0)
        ci_low = row.get("ci95_lower", 0)
        ci_high = row.get("ci95_upper", 0)
        p_bs = row.get("p_bootstrap", 1)
        p_mc = row.get("p_mcnemar", 1)
        sig = row.get("significance", "n.s.")

        # 유의한 결과 볼드 처리
        if sig != "n.s.":
            sig_str = r"\textbf{" + sig + "}"
        else:
            sig_str = sig

        p_bs_str = f"{p_bs:.1e}" if p_bs < 0.001 else f"{p_bs:.4f}"
        p_mc_str = f"{p_mc:.1e}" if p_mc < 0.001 else f"{p_mc:.4f}"

        lines.append(
            f"  vs.\\ {model_b} & "
            f"{delta_f1:+.4f} & "
            f"[{ci_low:.4f}, {ci_high:.4f}] & "
            f"{p_bs_str} & "
            f"{p_mc_str} & "
            f"{sig_str} \\\\"
        )

    lines.append(r"\hline\hline")
    lines.append(r"\multicolumn{6}{l}{\scriptsize * $p<0.05$, ** $p<0.01$, "
                 r"*** $p<0.001$ (Bonferroni corrected)} \\")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[Saved] {output_path}")


# ======================================================================
#  Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Statistical Significance Test — PSAN vs Baselines"
    )
    parser.add_argument(
        "--eval_dir", type=str, default="eval_results",
        help="Directory containing *_predictions.csv files",
    )
    parser.add_argument(
        "--proposed", type=str, default=PROPOSED_TAG,
        help="Model tag of the proposed model",
    )
    parser.add_argument(
        "--n_bootstrap", type=int, default=10000,
        help="Number of bootstrap iterations",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output_dir", type=str, default="significance_results",
        help="Output directory",
    )
    parser.add_argument(
        "--num_classes", type=int, default=11,
        help="Number of classes",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance level before Bonferroni correction",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 1. 파일 검색 ----
    print("=" * 70)
    print(" Statistical Significance Test")
    print("=" * 70)
    csv_map = discover_prediction_csvs(args.eval_dir)
    print(f"\n[Discovery] Found {len(csv_map)} prediction CSVs in '{args.eval_dir}':")
    for tag, path in csv_map.items():
        name = ALL_MODELS.get(tag, tag)
        print(f"  {tag:30s} → {name}")

    if args.proposed not in csv_map:
        print(f"\n[ERROR] Proposed model '{args.proposed}' not found!")
        print(f"  Available: {list(csv_map.keys())}")
        print(f"\n  Tip: --proposed 인자에 올바른 model_tag를 지정하세요.")
        print(f"       예: --proposed psan_tiny_bw02")
        return

    # ---- 2. 데이터 로드 ----
    print(f"\n[Loading] proposed = {args.proposed}")
    proposed_df = load_predictions(csv_map[args.proposed])
    proposed_correct = proposed_df["correct"].values.astype(bool)
    proposed_pred = proposed_df["pred_label"].values
    proposed_probs = extract_probs(proposed_df)
    y_true = proposed_df["true_label"].values
    n_samples = len(y_true)
    print(f"  N = {n_samples} samples")

    # 비교 대상 필터링
    compare_tags = [t for t in csv_map if t != args.proposed]
    n_comparisons = len(compare_tags)
    bonf_n = n_comparisons
    print(f"  Comparisons: {n_comparisons} (Bonferroni factor = {bonf_n})")

    # ---- 3. 검정 실행 ----
    mcnemar_rows = []
    bootstrap_rows = []
    summary_rows = []

    for tag in compare_tags:
        name = ALL_MODELS.get(tag, tag)
        print(f"\n{'─' * 50}")
        print(f"  {PROPOSED_NAME}  vs  {name}")
        print(f"{'─' * 50}")

        comp_df = load_predictions(csv_map[tag])

        # 샘플 정합성 체크
        if len(comp_df) != n_samples:
            print(f"  [SKIP] sample count mismatch: {len(comp_df)} vs {n_samples}")
            continue

        # 경로 순서 동일 여부 확인
        if not (comp_df["path"].values == proposed_df["path"].values).all():
            print("  [WARN] path order differs, re-aligning by path...")
            comp_df = comp_df.set_index("path").loc[proposed_df["path"]].reset_index()

        comp_correct = comp_df["correct"].values.astype(bool)
        comp_pred = comp_df["pred_label"].values
        comp_probs = extract_probs(comp_df)

        # ---- McNemar ----
        mc = mcnemar_test(proposed_correct, comp_correct)
        mc["Model A"] = PROPOSED_NAME
        mc["Model B"] = name
        mc["Model A tag"] = args.proposed
        mc["Model B tag"] = tag
        mc["bonferroni_p"] = min(mc["p_value_exact"] * bonf_n, 1.0)
        mc["significant"] = significance_stars(mc["p_value_exact"], bonf_n)

        print(f"  McNemar: discordant={mc['discordant_pairs']}  "
              f"p_exact={mc['p_value_exact']:.6f}  "
              f"p_bonf={mc['bonferroni_p']:.6f}  "
              f"→ {mc['significant']}")

        mcnemar_rows.append(mc)

        # ---- Paired Bootstrap ----
        bs = paired_bootstrap_test(
            y_true, proposed_pred, comp_pred,
            proposed_probs, comp_probs,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            num_classes=args.num_classes,
        )
        bs_row = {
            "Model A": PROPOSED_NAME,
            "Model B": name,
            "Model A tag": args.proposed,
            "Model B tag": tag,
            "model_b_name": name,
            "metrics": bs,
        }
        bootstrap_rows.append(bs_row)

        mf1 = bs.get("macro_f1", {})
        print(f"  Bootstrap (Macro F1): Δ={mf1.get('observed_diff',0):+.4f}  "
              f"95% CI=[{mf1.get('ci95_lower',0):.4f}, {mf1.get('ci95_upper',0):.4f}]  "
              f"p={mf1.get('p_value',1):.6f}")

        # ---- Summary row ----
        summary_rows.append({
            "Model A": PROPOSED_NAME,
            "Model B": name,
            "delta_accuracy": bs.get("accuracy", {}).get("observed_diff", np.nan),
            "delta_balanced_acc": bs.get("balanced_accuracy", {}).get("observed_diff", np.nan),
            "delta_macro_f1": mf1.get("observed_diff", np.nan),
            "ci95_lower": mf1.get("ci95_lower", np.nan),
            "ci95_upper": mf1.get("ci95_upper", np.nan),
            "p_bootstrap": mf1.get("p_value", np.nan),
            "p_mcnemar": mc["p_value_exact"],
            "bonferroni_p_bootstrap": min(mf1.get("p_value", 1) * bonf_n, 1.0),
            "bonferroni_p_mcnemar": mc["bonferroni_p"],
            "significance": significance_stars(
                min(mf1.get("p_value", 1), mc["p_value_exact"]), bonf_n
            ),
            "a_better_pct": mf1.get("a_better_pct", np.nan),
        })

    # ---- 4. 결과 저장 ----
    print(f"\n{'=' * 70}")
    print(" Saving results")
    print(f"{'=' * 70}")

    # McNemar CSV
    mc_df = pd.DataFrame(mcnemar_rows)
    mc_csv = os.path.join(args.output_dir, "significance_mcnemar.csv")
    mc_df.to_csv(mc_csv, index=False)
    print(f"[Saved] {mc_csv}")

    # Bootstrap CSV (flatten)
    bs_flat_rows = []
    for br in bootstrap_rows:
        for metric_name, metric_vals in br["metrics"].items():
            row = {
                "Model A": br["Model A"],
                "Model B": br["Model B"],
                "metric": metric_name,
                **metric_vals,
                "bonferroni_p": min(metric_vals["p_value"] * bonf_n, 1.0),
                "significant": significance_stars(metric_vals["p_value"], bonf_n),
            }
            bs_flat_rows.append(row)
    bs_df = pd.DataFrame(bs_flat_rows)
    bs_csv = os.path.join(args.output_dir, "significance_paired_bootstrap.csv")
    bs_df.to_csv(bs_csv, index=False)
    print(f"[Saved] {bs_csv}")

    # Summary CSV
    summary_df = pd.DataFrame(summary_rows)
    sum_csv = os.path.join(args.output_dir, "significance_summary.csv")
    summary_df.to_csv(sum_csv, index=False)
    print(f"[Saved] {sum_csv}")

    # LaTeX
    # 베이스라인만 분리하여 LaTeX 테이블 생성
    baseline_tags = set(BASELINE_MODELS.keys())
    baseline_summary = summary_df[
        summary_df["Model B"].isin(BASELINE_MODELS.values())
    ].copy()
    if len(baseline_summary) > 0:
        tex_path = os.path.join(args.output_dir, "significance_baselines.tex")
        generate_latex_table(baseline_summary, tex_path)

    # Ablation LaTeX
    ablation_summary = summary_df[
        summary_df["Model B"].isin(ABLATION_MODELS.values())
    ].copy()
    if len(ablation_summary) > 0:
        tex_path_abl = os.path.join(args.output_dir, "significance_ablations.tex")
        generate_latex_table(ablation_summary, tex_path_abl)

    # ---- 5. 시각화 ----
    # McNemar heatmap (baselines only)
    if len(mc_df) > 0:
        mc_baselines = mc_df[mc_df["Model B"].isin(BASELINE_MODELS.values())]
        if len(mc_baselines) > 0:
            heatmap_path = os.path.join(
                args.output_dir, "significance_mcnemar_heatmap.png"
            )
            try:
                plot_pvalue_heatmap(mc_baselines, heatmap_path,
                                   PROPOSED_NAME, n_comparisons)
            except Exception as e:
                print(f"[WARN] Heatmap failed: {e}")

    # Forest plot (baselines)
    baseline_bs = [br for br in bootstrap_rows
                   if br["Model B"] in BASELINE_MODELS.values()]
    if baseline_bs:
        forest_path = os.path.join(
            args.output_dir, "significance_forest_macro_f1.png"
        )
        plot_bootstrap_ci(baseline_bs, forest_path, PROPOSED_NAME, "macro_f1")

        # Also for AUC
        forest_auc = os.path.join(
            args.output_dir, "significance_forest_macro_auc.png"
        )
        plot_bootstrap_ci(baseline_bs, forest_auc, PROPOSED_NAME, "macro_auc")

    # ---- 6. 콘솔 요약 출력 ----
    print(f"\n{'=' * 70}")
    print(" SUMMARY: Statistical Significance Results")
    print(f"{'=' * 70}")
    print(f"  Proposed model : {PROPOSED_NAME}")
    print(f"  Bootstrap iter : {args.n_bootstrap}")
    print(f"  Alpha          : {args.alpha}")
    print(f"  Bonferroni N   : {bonf_n}")
    print(f"  Bonferroni α'  : {args.alpha / bonf_n:.6f}")
    print()

    print(f"{'Model B':<25s} {'ΔF1':>8s} {'95% CI':>22s} "
          f"{'p(BS)':>10s} {'p(McN)':>10s} {'Sig':>5s}")
    print("─" * 85)
    for _, row in summary_df.iterrows():
        model_b = row["Model B"]
        delta = row.get("delta_macro_f1", 0)
        ci_l = row.get("ci95_lower", 0)
        ci_h = row.get("ci95_upper", 0)
        p_bs = row.get("p_bootstrap", 1)
        p_mc = row.get("p_mcnemar", 1)
        sig = row.get("significance", "n.s.")
        print(
            f"  {model_b:<23s} {delta:>+.4f}  "
            f"[{ci_l:>+.4f}, {ci_h:>+.4f}]  "
            f"{p_bs:>10.6f}  {p_mc:>10.6f}  {sig:>5s}"
        )

    # ---- 7. JSON 저장 ----
    json_summary = {
        "config": {
            "proposed": args.proposed,
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "alpha": args.alpha,
            "bonferroni_n": bonf_n,
            "n_samples": int(n_samples),
        },
        "results": summary_rows,
    }
    json_path = os.path.join(args.output_dir, "significance_results.json")
    with open(json_path, "w") as f:
        json.dump(json_summary, f, indent=2, default=str)
    print(f"\n[Saved] {json_path}")

    print(f"\n[Done] All results saved to '{args.output_dir}/'")


if __name__ == "__main__":
    main()
