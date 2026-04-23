"""
datasets2/ 를 train/val/test 8:1:1로 클래스별 층화 분할.

동작:
  - 각 클래스 폴더의 .jpg를 파일명으로 정렬 후 시드 기반 셔플.
  - [0 : 0.8n) → train, [0.8n : 0.9n) → val, [0.9n : n) → test.
  - .jpg와 동명의 .json이 존재하면 같은 split으로 함께 복사/이동.
  - 출력 구조:
      <out>/train/<Class>/<files>
      <out>/val/<Class>/<files>
      <out>/test/<Class>/<files>
  - split_manifest.csv 로 (class, split, filename) 기록 — 재현성.

사용법:
  python split_dataset.py                                 # copy, seed=42
  python split_dataset.py --move                          # 파일 이동 (원본 제거)
  python split_dataset.py --seed 0 --out dataset_split    # 시드/출력 경로 변경
  python split_dataset.py --dry-run                       # 계획만 출력

안전장치:
  - 기본은 copy (원본 보존). --move는 명시적.
  - 출력 디렉토리에 동일 이름 파일 존재 시 기본적으로 건너뜀 (--overwrite로 덮어쓰기).
  - 이미지 개수 0인 클래스는 경고 후 스킵.
"""

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path


SPLITS = ("train", "val", "test")


def split_indices(n: int) -> dict[str, tuple[int, int]]:
    """n 개 샘플에 대한 split 경계 [start, end). 8:1:1."""
    n_train = int(n * 0.8)
    n_val = int(n * 0.9) - n_train
    return {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n),
    }


def gather_class_files(class_dir: Path) -> list[Path]:
    return sorted(class_dir.glob("*.jpg"))


def transfer(src: Path, dst: Path, move: bool, overwrite: bool, dry_run: bool) -> str:
    """한 파일을 전송. 반환: 'copied' | 'moved' | 'skipped' | 'planned'."""
    if dst.exists() and not overwrite:
        return "skipped"
    if dry_run:
        return "planned"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(src), str(dst))
        return "moved"
    shutil.copy2(str(src), str(dst))
    return "copied"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).parent / "datasets2",
                        help="원본 데이터셋 루트")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "dataset_split",
                        help="split 결과 출력 루트")
    parser.add_argument("--seed", type=int, default=42,
                        help="셔플 시드 (기본 42)")
    parser.add_argument("--move", action="store_true",
                        help="copy 대신 move (원본 제거)")
    parser.add_argument("--overwrite", action="store_true",
                        help="출력 파일이 이미 존재해도 덮어쓰기")
    parser.add_argument("--dry-run", action="store_true",
                        help="실제 전송 없이 계획만 출력")
    args = parser.parse_args()

    if not args.root.exists():
        raise SystemExit(f"원본 루트를 찾을 수 없습니다: {args.root}")

    mode = "MOVE" if args.move else "COPY"
    if args.dry_run:
        mode = f"DRY-RUN ({mode})"
    print(f"[{mode}] {args.root} -> {args.out} | seed={args.seed} | 8:1:1\n")

    class_dirs = sorted(p for p in args.root.iterdir() if p.is_dir())
    manifest: list[tuple[str, str, str]] = []  # (class, split, filename)
    per_class_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"train": 0, "val": 0, "test": 0, "total": 0}
    )
    op_counts: dict[str, int] = defaultdict(int)

    rng = random.Random(args.seed)

    for cls_dir in class_dirs:
        jpgs = gather_class_files(cls_dir)
        n = len(jpgs)
        if n == 0:
            print(f"  [skip] {cls_dir.name}: jpg 0개")
            continue

        order = list(range(n))
        rng.shuffle(order)
        shuffled = [jpgs[i] for i in order]
        bounds = split_indices(n)

        per_class_counts[cls_dir.name]["total"] = n
        for split, (s, e) in bounds.items():
            per_class_counts[cls_dir.name][split] = e - s
            for src in shuffled[s:e]:
                dst_dir = args.out / split / cls_dir.name
                dst = dst_dir / src.name
                result = transfer(src, dst, args.move, args.overwrite, args.dry_run)
                op_counts[result] += 1

                # 동명의 json 동반 이동/복사
                src_json = src.with_suffix(".json")
                if src_json.exists():
                    dst_json = dst_dir / src_json.name
                    jres = transfer(src_json, dst_json, args.move, args.overwrite, args.dry_run)
                    op_counts[f"json_{jres}"] += 1

                manifest.append((cls_dir.name, split, src.name))

        print(f"  {cls_dir.name}: total={n} "
              f"train={bounds['train'][1]-bounds['train'][0]} "
              f"val={bounds['val'][1]-bounds['val'][0]} "
              f"test={bounds['test'][1]-bounds['test'][0]}")

    # Manifest 기록
    if not args.dry_run:
        args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "split_manifest.csv"
    if not args.dry_run:
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["class", "split", "filename"])
            w.writerows(manifest)

    summary_path = args.out / "split_summary.csv"
    if not args.dry_run:
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["class", "total", "train", "val", "test"])
            for cls in sorted(per_class_counts):
                c = per_class_counts[cls]
                w.writerow([cls, c["total"], c["train"], c["val"], c["test"]])

    # 전체 요약
    totals = {"train": 0, "val": 0, "test": 0, "total": 0}
    for c in per_class_counts.values():
        for k in totals:
            totals[k] += c[k]
    print(f"\n전체: total={totals['total']} "
          f"train={totals['train']} val={totals['val']} test={totals['test']}")

    print(f"\n작업 건수: {dict(op_counts)}")
    if not args.dry_run:
        print(f"\n저장됨:\n  {manifest_path}\n  {summary_path}")
    else:
        print("\n(dry-run — 실제 전송/기록 없음. 제거하려면 --dry-run 빼고 재실행)")


if __name__ == "__main__":
    main()
