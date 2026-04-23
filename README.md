# PSAN-Parametric-Spectral-Atom-Network-with-Frequency-Attribution

# Canine Ocular Disease Dataset
 
Korean AI Hub 반려동물 안구질환 데이터셋의 전처리 요약.
 
> **Source:** [AI Hub — 반려동물 안구질환 데이터](https://www.aihub.or.kr/)
 
## Deduplication Pipeline
 
원본 데이터셋에 대해 2단계 중복 제거를 수행하였다.
 
1. **MD5 hash matching** — 바이트 단위 완전 동일 이미지 제거
2. **Perceptual hashing (pHash)** — Hamming distance ≤ 5 기준 근사 중복 제거
## Dataset Statistics
 
### Overall
 
|  | Images | Ratio |
|---|---:|---:|
| Before deduplication | 152,937 | 100.0% |
| After deduplication | 91,166 | 59.6% |
| **Removed** | **61,771** | **40.4%** |
 
### Per-Class Breakdown
 
| Class | Before | After | Removed | Removal Rate |
|---|---:|---:|---:|---:|
| Normal | 44,932 | 30,465 | 14,467 | 32.2% |
| Cataract | 23,216 | 19,737 | 3,479 | 15.0% |
| Ulcerative keratitis | 15,468 | 7,541 | 7,927 | 51.2% |
| Blepharitis | 7,738 | 7,463 | 275 | 3.6% |
| Conjunctivitis | 10,799 | 7,184 | 3,615 | 33.5% |
| Entropion | 8,667 | 4,685 | 3,982 | 45.9% |
| Non-ulcerative keratitis | 10,800 | 3,556 | 7,244 | 67.1% |
| Epiphora | 7,211 | 3,320 | 3,891 | 54.0% |
| Nuclear sclerosis | 10,798 | 3,049 | 7,749 | 71.8% |
| Pigmentary keratitis | 7,922 | 2,224 | 5,698 | 71.9% |
| Eyelid tumor | 5,386 | 1,942 | 3,444 | 63.9% |
| **Total** | **152,937** | **91,166** | **61,771** | **40.4%** |
 
### Train / Val / Test Split (After Deduplication)
 
Stratified sampling으로 80 / 10 / 10 비율 분할.
 
| Class | Total | Train | Val | Test |
|---|---:|---:|---:|---:|
| Normal | 30,465 | 24,372 | 3,046 | 3,047 |
| Cataract | 19,737 | 15,789 | 1,974 | 1,974 |
| Ulcerative keratitis | 7,541 | 6,032 | 754 | 755 |
| Blepharitis | 7,463 | 5,970 | 746 | 747 |
| Conjunctivitis | 7,184 | 5,747 | 718 | 719 |
| Entropion | 4,685 | 3,748 | 468 | 469 |
| Non-ulcerative keratitis | 3,556 | 2,844 | 356 | 356 |
| Epiphora | 3,320 | 2,656 | 332 | 332 |
| Nuclear sclerosis | 3,049 | 2,439 | 305 | 305 |
| Pigmentary keratitis | 2,224 | 1,779 | 222 | 223 |
| Eyelid tumor | 1,942 | 1,553 | 194 | 195 |
| **Total** | **91,166** | **72,929** | **9,115** | **9,122** |
 
## Notes
 
- 클래스 간 최대 불균형 비율: **15.7 : 1** (Normal 30,465 vs Eyelid tumor 1,942)
- 중복 제거율이 가장 높은 클래스: Pigmentary keratitis (71.9%), Nuclear sclerosis (71.8%)
- 중복 제거율이 가장 낮은 클래스: Blepharitis (3.6%), Cataract (15.0%)
- 학습 시 클래스 불균형 보정을 위해 inverse-square-root 가중 샘플링 적용: $w_c = 1 / \sqrt{N_c}$
