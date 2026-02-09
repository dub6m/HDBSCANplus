# HDBSCANplus

`HDBSCANplus` is a **self-converging wrapper** around [`hdbscan`](https://github.com/scikit-learn-contrib/hdbscan) for embedding clustering.

Instead of running one static HDBSCAN configuration, it:

1. Prepares embeddings (optionally L2-normalized)
2. Runs HDBSCAN for multiple parameter candidates
3. Scores each trial using a blended quality objective
4. Expands search around promising candidates until it reaches a trial budget

The result is a practical “autotuned HDBSCAN” workflow for dense vector data.

---

## What problem this solves

Choosing good HDBSCAN settings (especially `min_cluster_size`) is often dataset-specific. `HDBSCANplus` automates this search and uses internal quality signals to rank outcomes:

- **DBCV** for density-based cluster validity
- **BIC-style compactness/separation signal** for model fit
- **Penalties** for noisy, degenerate, or over-fragmented solutions

This lets you start from reasonable defaults and still get adaptive behavior per dataset.

---

## Features

- ✅ Local search over `minClusterSize` and `clusterSelectionEpsilon`
- ✅ Robust scoring pipeline with gated DBCV/BIC blending
- ✅ Penalty terms for:
  - high noise rate
  - single-cluster collapse
  - tiny clusters
  - mixed-cluster splitting tendency
- ✅ Works directly with NumPy embedding matrices
- ✅ Returns rich diagnostics (`scoreDetails`, `clusterStats`, full trial history)
- ✅ Handles edge cases (empty input, very small datasets, trial failures)

---

## Installation

### Requirements

- Python 3.9+
- `numpy`
- `hdbscan`
- `scikit-learn` *(optional but recommended for fast normalization)*

### Install dependencies

```bash
pip install numpy hdbscan scikit-learn
```

---

## Quick start

```python
import numpy as np
from HDBSCANplus import HDBSCANplus

rng = np.random.default_rng(42)
a = rng.normal(size=(200, 32)) * 0.4 + 0.0
b = rng.normal(size=(200, 32)) * 0.4 + 3.0
x = np.vstack([a, b]).astype(np.float32)

clusterer = HDBSCANplus(
    metric="cosine",           # metric argument is accepted, run metric is euclidean on normalized vectors
    normalizeVectors=True,
    maxTrials=60,
)

result = clusterer.fitPredict(x)

print("Best score:", result.bestScore)
print("Best params:", result.bestParams)
print("Cluster stats:", result.clusterStats)
print("Unique labels:", sorted(set(result.labels.tolist())))
```

---

## API overview

### `HDBSCANplus(...)`

Main class.

#### Important constructor parameters

- `normalizeVectors=True`
  - L2-normalizes each embedding row before clustering.
- `clusterSelectionMethod="eom"`
  - Passed to HDBSCAN's `cluster_selection_method`.
- `maxTrials=120`
  - Maximum number of parameter trials to evaluate.
- `minClusterSizeRange=(3, 80)`
  - Search range for `min_cluster_size`.
- `clusterSelectionEpsilonRange=(0.0, 0.0)`
  - Search range for `cluster_selection_epsilon`.
- `epsilonStep=0.05`
  - Neighbor-search step for epsilon.
- `dbcvGate=0.4`
  - If normalized DBCV is below this gate, score uses DBCV only.
- `alpha=0.7`
  - Blend ratio once above gate: `alpha * DBCV + (1-alpha) * BICScore`.
- `noisePenaltyWeight=0.6`, `singleClusterPenalty=0.4`, `tinyClusterPenaltyWeight=0.3`
  - Penalty weights.
- `minUsefulClusterSize=4`
  - Cluster sizes below this count as “tiny”.
- `lambdaMax=1.0`
  - Upper bound for mixed-cluster penalty scaling.
- `expandTopK=5`
  - Number of best trials used to expand local search neighborhood.

### `fitPredict(embeddings)`

Runs the full optimization pipeline and returns `HdbscanPlusResult`.

Input:

- 2D array-like (`n_samples x n_features`)

Output fields:

- `labels: np.ndarray`
- `probabilities: np.ndarray`
- `bestScore: float`
- `bestParams: dict`
- `scoreDetails: dict`
- `clusterStats: dict`
- `tried: list` *(all evaluated trials with details)*

---

## Scoring logic (high level)

For each trial:

1. Run HDBSCAN with candidate params
2. Compute:
   - `dbcvRaw` and normalized `dbcvNorm` in `[0, 1]`
   - `bicScore` in `[0, 1]` (relative improvement over single-cluster null)
3. Base score:
   - If `dbcvNorm < dbcvGate`: `base = dbcvNorm`
   - Else: `base = alpha*dbcvNorm + (1-alpha)*bicScore`
4. Subtract penalties:
   - sanity penalty (noise, single-cluster, tiny-cluster)
   - mixed-cluster penalty scaled by `lambdaMax * dbcvNorm`

Final score:

```text
score = baseScore - sanityPenalty - mixedLambda * mixedPenalty
```

Higher is better.

---

## Search strategy

`HDBSCANplus` does a bounded local search:

- Start from a base `minClusterSize ≈ sqrt(n)` and epsilon baseline
- Seed frontier with smaller/base/larger `minClusterSize`
- Evaluate candidates and keep ranked trials
- Repeatedly expand neighborhoods around top results (`expandTopK`)
- Stop when frontier is exhausted or `maxTrials` reached

This provides better coverage than one-shot tuning while staying computationally bounded.

---

## Behavior in edge cases

- **Empty input (`n=0`)**
  - Returns an empty result object with score `-1.0`.
- **Very small datasets (`n<5`)**
  - Uses default small-N parameters and a single trial.
- **Runtime failure inside one trial**
  - Trial is retained with score `-1.0`, all-noise labels, and captured error text.

---

## Practical tips

- Start with defaults, then widen `minClusterSizeRange` if cluster scales vary a lot.
- If you expect fine-grained density structure, allow epsilon search:
  - `clusterSelectionEpsilonRange=(0.0, 0.3)` and `epsilonStep=0.02`.
- If results are too noisy, increase `noisePenaltyWeight`.
- If it collapses into one cluster too often, increase `singleClusterPenalty`.
- If tiny fragments appear, increase `tinyClusterPenaltyWeight` or `minUsefulClusterSize`.
- Raise `maxTrials` and `expandTopK` for deeper tuning (with higher runtime).

---

## Notes on metric handling

Current implementation always runs HDBSCAN with Euclidean distance on prepared data. With default normalization enabled, this gives cosine-like behavior for embeddings in many workflows.

If you rely on a specific metric path, verify behavior in `prepareData()` and `_runHdbscan()` before extending.

---

## Repository contents

- `HDBSCANplus.py` — implementation, result dataclass, and runnable demo in `__main__`.

---

## License

No license file is currently included in this repository. Add one if you plan to distribute this project.
