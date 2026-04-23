"""
  python count_classes.py
  python count_classes.py --root OriginalDataset --out-dir .
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def count_jpgs(root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for p in sorted(root.iterdir()):
        if p.is_dir():
            counts[p.name] = sum(1 for _ in p.glob("*.jpg"))
    return counts


def save_csv(counts: dict[str, int], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "count"])
        for cls, n in sorted(counts.items(), key=lambda x: -x[1]):
            w.writerow([cls, n])


def save_plot(counts: dict[str, int], path: Path) -> None:
    items = sorted(counts.items(), key=lambda x: -x[1])
    classes = [k for k, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(classes, values, color="steelblue", edgecolor="black")
    ax.set_xlabel("Class")
    ax.set_ylabel("Image count")
    ax.set_title(f"class distribution (total={sum(values):,})")
    ax.tick_params(axis="x", rotation=35)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")

    ymax = max(values) if values else 0
    ax.set_ylim(0, ymax * 1.12)
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v:,}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).parent / "datasets2",
                        help="데이터셋 루트 디렉토리")
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).parent,
                        help="csv/png 저장 경로")
    args = parser.parse_args()

    if not args.root.exists():
        raise SystemExit(f"루트를 찾을 수 없습니다: {args.root}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    counts = count_jpgs(args.root)
    if not counts:
        raise SystemExit(f"클래스 폴더를 찾지 못했습니다: {args.root}")

    csv_path = args.out_dir / "class_counts(dedup).csv"
    png_path = args.out_dir / "class_counts(dedup).png"
    save_csv(counts, csv_path)
    save_plot(counts, png_path)

    print(f"클래스 수: {len(counts)}, 총 이미지: {sum(counts.values()):,}")
    for cls, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {n:,}")
    print(f"\n저장됨:\n  {csv_path}\n  {png_path}")


if __name__ == "__main__":
    main()
