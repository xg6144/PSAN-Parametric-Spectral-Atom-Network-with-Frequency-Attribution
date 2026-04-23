"""
OriginalDataset 폴더 내 클래스(질병) 간 중복/근사 중복 이미지 탐지 & 제거.

2단계 파이프라인 (표준):
  Stage 1: MD5 — 바이트 단위 완전 동일 파일 탐지 (빠른 1차 필터).
  Stage 2: pHash (perceptual hash) + Hamming 거리 — 리사이즈/재압축/약한 편집 등
           시각적으로 동일/거의 동일한 이미지까지 탐지. 기본 임계값은 5.

중복 군집 구성:
  - Stage 1에서 동일 MD5끼리 먼저 묶고, 그 그룹의 대표 하나씩만 Stage 2로 투입.
  - Stage 2: pHash 해밍거리 ≤ threshold 쌍을 LSH로 찾아 Union-Find로 군집화.
  - 주의: UF는 transitive closure라 A~B, B~C 쌍만 가까워도 A-C가 threshold 이상으로
    벌어진 상태로 같은 군집에 묶일 수 있음(chain 효과). 의료 영상 라벨 노이즈 제거
    맥락에서 이는 정상 데이터 과삭제 위험. 본 스크립트는:
      (a) 모든 군집의 diameter(내부 최대 쌍거리)를 계산해 로그/분포 출력
      (b) --strict-cluster 지정 시, diameter > threshold 인 군집은 UF 병합을 취소하고
          각 MD5 그룹으로 되돌림 (= 군집이 실제로 clique 일 때만 유지).

삭제 정책:
  기본: cross-class 군집은 모든 복사본 제거 — 라벨 모호성.
  --keep-one: 알파벳순 첫 클래스의 한 장만 남기고 나머지 제거.
  같은 클래스 내부 중복은 항상 한 장만 남기고 제거.

재현성 로그 (항상 기록):
  - dedup_log_<timestamp>.csv : 모든 중복 군집의 각 파일에 대한 kept/deleted 상태
  - dedup_log_<timestamp>.json: 실행 메타데이터 (파라미터, 시각, git commit, 통계)

안전장치:
  - 기본은 dry-run, 실제 삭제는 --execute.
  - .jpg 삭제 시 동명의 .json도 함께 삭제.
  - 손상된 이미지는 경고 후 스킵.

사용법:
  python remove_duplicates.py                             # dry-run, MD5+pHash(5)
  python remove_duplicates.py --strict-cluster            # clique 기반, 과삭제 방지
  python remove_duplicates.py --phash-threshold 3         # 더 엄격
  python remove_duplicates.py --no-phash                  # Stage 1만 수행
  python remove_duplicates.py --keep-one --execute        # 한 장 남기고 실제 삭제

의존성: imagehash, Pillow, numpy
"""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

try:
    import imagehash
except ImportError:
    sys.exit("imagehash 패키지가 필요합니다: pip install imagehash")


DATASET_DIR = Path(__file__).parent / "OriginalDataset"
LOG_DIR = Path(__file__).parent / "dedup_logs"
CHUNK = 1024 * 1024
PHASH_SIZE = 8  # 8x8 DCT 저주파 -> 64 bits
HASH_BITS = PHASH_SIZE * PHASH_SIZE


# ---------------- Stage 1: MD5 ----------------

def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_md5(root: Path) -> dict[str, list[Path]]:
    hashes: dict[str, list[Path]] = defaultdict(list)
    class_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    total = 0
    for cls in class_dirs:
        files = sorted(cls.glob("*.jpg"))
        print(f"  [MD5] {cls.name}: {len(files)}")
        for f in files:
            hashes[md5sum(f)].append(f)
            total += 1
    print(f"  총 {total}개 jpg, 고유 MD5 {len(hashes)}개")
    return hashes


# ---------------- Stage 2: pHash ----------------

def compute_phash(path_str: str) -> tuple[str, int | None]:
    """pHash를 uint64로 반환. 실패 시 None.

    imagehash.phash 내부가 grayscale 변환을 처리하므로 외부 convert 불필요.
    """
    try:
        with Image.open(path_str) as img:
            h = imagehash.phash(img, hash_size=PHASH_SIZE)
        # h.hash는 (N,N) bool ndarray. flatten 순서는 row-major.
        # 주: pigeonhole 정리는 비트 분할 방식과 무관하게 성립하므로
        #     DCT 계수의 공간적 의미를 보존할 필요 없음.
        bits = h.hash.flatten()
        value = 0
        for b in bits:
            value = (value << 1) | int(b)
        return path_str, value
    except (UnidentifiedImageError, OSError, ValueError) as e:
        warnings.warn(f"pHash 실패: {path_str} ({e})")
        return path_str, None


def compute_phashes_parallel(paths: list[Path], workers: int) -> dict[Path, int]:
    result: dict[Path, int] = {}
    failed = 0
    if workers <= 1:
        for p in paths:
            _, v = compute_phash(str(p))
            if v is None:
                failed += 1
            else:
                result[p] = v
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(compute_phash, str(p)) for p in paths]
            done = 0
            for fut in as_completed(futures):
                path_str, v = fut.result()
                if v is None:
                    failed += 1
                else:
                    result[Path(path_str)] = v
                done += 1
                if done % 5000 == 0:
                    print(f"  [pHash] {done}/{len(paths)}")
    if failed:
        print(f"  경고: {failed}개 파일 pHash 실패 (스킵)")
    return result


# ---------------- Union-Find ----------------

class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1


# ---------------- Near-duplicate search (segment LSH) ----------------

def segment_ranges(total_bits: int, n_segs: int) -> list[tuple[int, int]]:
    """total_bits를 n_segs로 거의 균등 분할. 반환: [(offset, size), ...]."""
    base = total_bits // n_segs
    extra = total_bits - base * n_segs
    ranges = []
    off = 0
    for i in range(n_segs):
        size = base + (1 if i < extra else 0)
        ranges.append((off, size))
        off += size
    return ranges


def popcount64(x: np.ndarray) -> np.ndarray:
    """uint64 배열의 bit popcount. dtype 안전성을 위해 입력을 uint64로 강제."""
    x = x.astype(np.uint64, copy=False)
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


def find_near_duplicate_pairs(hashes: np.ndarray, threshold: int) -> list[tuple[int, int]]:
    """hashes: uint64 배열. hamming ≤ threshold 인 (i,j) 쌍 리스트.

    pigeonhole: 64비트를 (threshold+1)개 세그먼트로 나누면, 해밍거리 ≤ threshold 인
    두 해시는 최소 하나의 세그먼트 값이 완전히 동일. 세그먼트 값으로 버킷팅 후 실제
    해밍거리 검사.
    """
    n = len(hashes)
    if n < 2 or threshold < 0:
        return []
    n_segs = threshold + 1
    ranges = segment_ranges(HASH_BITS, n_segs)

    seen: set[tuple[int, int]] = set()
    thr = np.uint64(threshold)

    for seg_idx, (off, size) in enumerate(ranges):
        mask = np.uint64((1 << size) - 1)
        shift = np.uint64(off)
        keys = (hashes >> shift) & mask
        buckets: dict[int, list[int]] = defaultdict(list)
        for i, k in enumerate(keys.tolist()):
            buckets[k].append(i)

        for idxs in buckets.values():
            if len(idxs) < 2:
                continue
            arr = np.array(idxs, dtype=np.int64)
            h_sub = hashes[arr]
            m = len(idxs)
            for a in range(m - 1):
                # XOR 결과가 uint64 유지되도록 명시적 호출
                xor = np.bitwise_xor(h_sub[a], h_sub[a + 1:])
                diffs = popcount64(xor)
                close = np.nonzero(diffs <= thr)[0]
                for rel in close.tolist():
                    i = int(arr[a])
                    j = int(arr[a + 1 + rel])
                    if i > j:
                        i, j = j, i
                    seen.add((i, j))

        if (seg_idx + 1) % max(1, n_segs // 4) == 0:
            print(f"  [pHash LSH] 세그먼트 {seg_idx + 1}/{n_segs} 처리, 누적 쌍 {len(seen)}")

    return list(seen)


# ---------------- Cluster diameter & strict validation ----------------

def cluster_diameter(rep_hashes: np.ndarray) -> int:
    """군집 내 대표 해시들의 최대 쌍거리. 크기 ≤ 1 이면 0."""
    m = len(rep_hashes)
    if m <= 1:
        return 0
    d_max = 0
    for a in range(m - 1):
        xor = np.bitwise_xor(rep_hashes[a], rep_hashes[a + 1:])
        d = int(popcount64(xor).max())
        if d > d_max:
            d_max = d
    return d_max


# ---------------- Cluster building ----------------

def build_clusters(md5_groups: dict[str, list[Path]],
                   phash_threshold: int,
                   workers: int,
                   strict: bool):
    """최종 중복 군집 + 진단 정보 반환.

    Returns: (clusters, diagnostics)
      clusters: list[list[Path]] — 각 원소는 크기 ≥ 2 중복 군집
      diagnostics: dict — diameter 분포 등
    """
    md5_clusters = [paths for paths in md5_groups.values()]
    reps = [paths[0] for paths in md5_clusters]

    diagnostics = {
        "md5_groups_total": len(md5_clusters),
        "md5_dup_groups": sum(1 for c in md5_clusters if len(c) >= 2),
        "phash_pairs": 0,
        "stage2_merged_components": 0,
        "strict_rejected_components": 0,
        "diameter_histogram": {},
        "large_cluster_warnings": [],
    }

    if phash_threshold < 0:
        return [c for c in md5_clusters if len(c) >= 2], diagnostics

    print(f"\n  Stage 2: {len(reps)}개 대표 이미지의 pHash 계산")
    phash_map = compute_phashes_parallel(reps, workers=workers)
    if not phash_map:
        print("  유효 pHash 0개 — Stage 2 결과 없음.")
        return [c for c in md5_clusters if len(c) >= 2], diagnostics

    valid_rep_indices = [i for i, r in enumerate(reps) if r in phash_map]
    hashes_u64 = np.array([phash_map[reps[i]] for i in valid_rep_indices], dtype=np.uint64)
    print(f"  유효 pHash {len(hashes_u64)}개 대상 근사 중복 탐색 (threshold={phash_threshold})")

    pairs_local = find_near_duplicate_pairs(hashes_u64, phash_threshold)
    diagnostics["phash_pairs"] = len(pairs_local)
    print(f"  pHash 근사 중복 쌍: {len(pairs_local)}")

    uf = UnionFind(len(reps))
    for i_local, j_local in pairs_local:
        uf.union(valid_rep_indices[i_local], valid_rep_indices[j_local])

    components: dict[int, list[int]] = defaultdict(list)
    for i in range(len(reps)):
        components[uf.find(i)].append(i)

    # Stage 2로 실제 병합된 (≥2 MD5 그룹) 컴포넌트 수
    merged_components = [c for c in components.values() if len(c) >= 2]
    diagnostics["stage2_merged_components"] = len(merged_components)

    # Diameter 진단 및 strict 모드 처리
    diameter_hist: dict[int, int] = defaultdict(int)
    rejected_comp_ids: set[int] = set()
    large_warnings: list[dict] = []

    for root, members in components.items():
        if len(members) <= 1:
            continue
        member_hashes = np.array(
            [phash_map[reps[m]] for m in members if reps[m] in phash_map],
            dtype=np.uint64,
        )
        if len(member_hashes) <= 1:
            continue
        diam = cluster_diameter(member_hashes)
        diameter_hist[diam] += 1

        total_files = sum(len(md5_clusters[m]) for m in members)
        if total_files > 10:
            large_warnings.append({
                "component_root": int(root),
                "md5_group_count": len(members),
                "total_files": total_files,
                "diameter": diam,
            })

        if strict and diam > phash_threshold:
            rejected_comp_ids.add(root)

    diagnostics["diameter_histogram"] = dict(sorted(diameter_hist.items()))
    diagnostics["large_cluster_warnings"] = large_warnings
    diagnostics["strict_rejected_components"] = len(rejected_comp_ids)

    if large_warnings:
        print(f"  경고: 파일 10개 초과 군집 {len(large_warnings)}개 — chain 효과 의심, 로그 확인 권장")
    if diameter_hist:
        print("  diameter 분포 (해밍거리):")
        for d, cnt in sorted(diameter_hist.items()):
            marker = " ← threshold 초과" if d > phash_threshold else ""
            print(f"    {d}: {cnt}{marker}")
    if strict and rejected_comp_ids:
        print(f"  [strict] diameter > threshold 로 {len(rejected_comp_ids)}개 컴포넌트 병합 취소")

    # 최종 군집 경로 평탄화
    final_clusters: list[list[Path]] = []
    for root, members in components.items():
        if strict and root in rejected_comp_ids:
            # 각 MD5 그룹을 개별 군집으로 되돌림
            for m in members:
                if len(md5_clusters[m]) >= 2:
                    final_clusters.append(md5_clusters[m])
        else:
            merged_paths: list[Path] = []
            for m in members:
                merged_paths.extend(md5_clusters[m])
            if len(merged_paths) >= 2:
                final_clusters.append(merged_paths)

    return final_clusters, diagnostics


# ---------------- Deletion planning ----------------

def plan_deletions(clusters: list[list[Path]], keep_one: bool) -> tuple[list[Path], dict]:
    to_delete: list[Path] = []
    cross_class_groups = 0
    intra_class_groups = 0
    largest = 0

    for paths in clusters:
        largest = max(largest, len(paths))
        classes = {p.parent.name for p in paths}

        if len(classes) >= 2:
            cross_class_groups += 1
            if keep_one:
                paths_sorted = sorted(paths, key=lambda p: (p.parent.name, p.name))
                to_delete.extend(paths_sorted[1:])
            else:
                to_delete.extend(paths)
        else:
            intra_class_groups += 1
            paths_sorted = sorted(paths, key=lambda p: p.name)
            to_delete.extend(paths_sorted[1:])

    stats = {
        "cross_class_clusters": cross_class_groups,
        "intra_class_clusters": intra_class_groups,
        "largest_cluster_size": largest,
        "jpg_to_delete": len(to_delete),
    }
    print(f"  cross-class 중복 군집: {cross_class_groups}")
    print(f"  class 내부 중복 군집: {intra_class_groups}")
    print(f"  가장 큰 군집 크기: {largest}")
    return to_delete, stats


def delete_pair(jpg: Path, execute: bool) -> tuple[int, int]:
    """.jpg와 동명 .json을 삭제 (execute=True일 때만 실제 삭제).

    Returns:
      (planned_count, removed_count)
        planned_count: 삭제 대상 파일 수 (jpg + 선택적 json)
        removed_count: 실제로 unlink 한 파일 수 (dry-run 이면 0)
    """
    planned = 0
    removed = 0
    json_path = jpg.with_suffix(".json")
    targets = [jpg] + ([json_path] if json_path.exists() else [])
    for t in targets:
        planned += 1
        if execute:
            t.unlink()
            removed += 1
    return planned, removed


# ---------------- Logging ----------------

def git_commit_hash(cwd: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def write_logs(log_dir: Path,
               timestamp: str,
               clusters: list[list[Path]],
               to_delete: list[Path],
               args: argparse.Namespace,
               diagnostics: dict,
               deletion_stats: dict,
               executed: bool) -> tuple[Path, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / f"dedup_log_{timestamp}.csv"
    meta_path = log_dir / f"dedup_log_{timestamp}.json"

    delete_set = {str(p) for p in to_delete}
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "cluster_id", "path", "class", "status",
            "cluster_size", "cluster_classes",
        ])
        for cid, paths in enumerate(clusters):
            classes = sorted({p.parent.name for p in paths})
            for p in paths:
                w.writerow([
                    cid,
                    str(p.relative_to(DATASET_DIR)),
                    p.parent.name,
                    "deleted" if str(p) in delete_set else "kept",
                    len(paths),
                    "|".join(classes),
                ])

    meta = {
        "timestamp": timestamp,
        "executed": executed,
        "dataset_dir": str(DATASET_DIR),
        "git_commit": git_commit_hash(Path(__file__).parent),
        "args": {
            "execute": args.execute,
            "keep_one": args.keep_one,
            "phash_threshold": args.phash_threshold,
            "no_phash": args.no_phash,
            "strict_cluster": args.strict_cluster,
            "workers": args.workers,
        },
        "phash_size_bits": HASH_BITS,
        "diagnostics": diagnostics,
        "deletion_stats": deletion_stats,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return csv_path, meta_path


# ---------------- Main ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true",
                        help="실제 삭제 수행 (미지정 시 dry-run)")
    parser.add_argument("--keep-one", action="store_true",
                        help="cross-class 중복 시 한 장만 남기고 삭제")
    parser.add_argument("--phash-threshold", type=int, default=5,
                        help="pHash 해밍거리 임계값 (기본 5). 작을수록 엄격.")
    parser.add_argument("--no-phash", action="store_true",
                        help="Stage 2(pHash) 스킵 — MD5만으로 중복 탐지")
    parser.add_argument("--strict-cluster", action="store_true",
                        help="군집의 diameter > threshold 면 병합을 취소 (clique 기반, 과삭제 방지)")
    parser.add_argument("--workers", type=int, default=4,
                        help="pHash 병렬 워커 수 (기본 4)")
    args = parser.parse_args()

    if not DATASET_DIR.exists():
        raise SystemExit(f"데이터셋 폴더를 찾을 수 없습니다: {DATASET_DIR}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "EXECUTE" if args.execute else "DRY-RUN"
    policy = "cross-class는 하나만 유지" if args.keep_one else "cross-class는 전부 제거"
    phase = "MD5만" if args.no_phash else f"MD5 + pHash(threshold={args.phash_threshold})"
    cluster_mode = "strict(clique)" if args.strict_cluster else "UF(transitive)"
    print(f"[{mode}] 정책: {policy} | 탐지: {phase} | 군집: {cluster_mode}\n")

    print("1) Stage 1: MD5 해시 수집")
    md5_groups = collect_md5(DATASET_DIR)

    print("\n2) 중복 군집 구성")
    phash_threshold = -1 if args.no_phash else args.phash_threshold
    clusters, diagnostics = build_clusters(
        md5_groups,
        phash_threshold=phash_threshold,
        workers=args.workers,
        strict=args.strict_cluster,
    )
    print(f"  최종 중복 군집 수: {len(clusters)}")

    print("\n3) 삭제 계획 수립")
    to_delete, deletion_stats = plan_deletions(clusters, keep_one=args.keep_one)
    print(f"  삭제 예정 jpg: {len(to_delete)}")

    per_class: dict[str, int] = defaultdict(int)
    for p in to_delete:
        per_class[p.parent.name] += 1
    if per_class:
        print("\n  클래스별 삭제 예정 수:")
        for cls, n in sorted(per_class.items()):
            print(f"    {cls}: {n}")

    # 로그 기록 (dry-run/execute 모두)
    csv_path, meta_path = write_logs(
        LOG_DIR, timestamp, clusters, to_delete,
        args, diagnostics, deletion_stats, executed=args.execute,
    )
    print(f"\n  로그 기록:")
    print(f"    {csv_path}")
    print(f"    {meta_path}")

    print("\n4) 삭제 수행" if args.execute else "\n4) (dry-run — 실제 삭제 없음)")
    total_planned = 0
    total_removed = 0
    for jpg in to_delete:
        p, r = delete_pair(jpg, execute=args.execute)
        total_planned += p
        total_removed += r

    if args.execute:
        print(f"\n완료. jpg+json 합계 {total_removed}개 삭제됨.")
    else:
        print(f"\n완료. jpg+json 합계 {total_planned}개 삭제 예정. 실제 삭제는 --execute.")


if __name__ == "__main__":
    main()
