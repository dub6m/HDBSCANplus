# Self-converging HDBSCAN wrapper (HDBSCAN+)
# Pipeline:
# 1) Embeddings (optional L2-normalized)
# 2) HDBSCAN (parameterized)
# 3) Score = DBCV floor + (optional) BIC ceiling - penalties
# 4) Local neighbor expansion search until no improvements

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import numpy as np

try:
    import hdbscan
    from hdbscan.validity import validity_index as dbcvValidityIndex
except Exception:
    hdbscan = None
    dbcvValidityIndex = None

try:
    from sklearn.preprocessing import normalize as skNormalize
except Exception:
    skNormalize = None


@dataclass
class HdbscanPlusResult:
    labels: np.ndarray
    probabilities: np.ndarray
    bestScore: float
    bestParams: dict
    scoreDetails: dict
    clusterStats: dict
    tried: list


class HDBSCANplus:
    def __init__(
        self,
        metric="euclidean",
        normalizeVectors=True,
        cosineViaNormalize=True,
        clusterSelectionMethod="eom",
        maxTrials=120,
        randomState=7,
        debug=False,
        minClusterSizeRange=(3, 80),
        clusterSelectionEpsilonRange=(0.0, 0.0),
        epsilonStep=0.05,
        dbcvGate=0.4,
        alpha=0.7,
        noisePenaltyWeight=0.6,
        singleClusterPenalty=0.4,
        tinyClusterPenaltyWeight=0.3,
        minUsefulClusterSize=4,
        lambdaMax=1.0,
        bicVarEps=1e-6,
        expandTopK=5,
        kmeansMaxIter=20,
    ):
        self.metric = metric
        self.normalizeVectors = bool(normalizeVectors)
        self.cosineViaNormalize = bool(cosineViaNormalize)
        self.clusterSelectionMethod = clusterSelectionMethod
        self.maxTrials = int(max(1, maxTrials))
        self.randomState = int(randomState)
        self.debug = bool(debug)

        self.minClusterSizeRange = (int(minClusterSizeRange[0]), int(minClusterSizeRange[1]))
        self.clusterSelectionEpsilonRange = (
            float(clusterSelectionEpsilonRange[0]),
            float(clusterSelectionEpsilonRange[1]),
        )

        self.epsilonStep = float(max(0.0, epsilonStep))

        self.dbcvGate = float(max(0.0, min(1.0, dbcvGate)))
        self.alpha = float(max(0.0, min(1.0, alpha)))

        self.noisePenaltyWeight = float(max(0.0, noisePenaltyWeight))
        self.singleClusterPenalty = float(max(0.0, singleClusterPenalty))
        self.tinyClusterPenaltyWeight = float(max(0.0, tinyClusterPenaltyWeight))
        self.minUsefulClusterSize = int(max(2, minUsefulClusterSize))
        self.lambdaMax = float(max(0.0, lambdaMax))
        self.bicVarEps = float(max(1e-12, bicVarEps))

        self.expandTopK = int(max(1, expandTopK))
        self.kmeansMaxIter = int(max(1, kmeansMaxIter))

        self.metricForRun = "euclidean"
        self.bestResult = None

        self._sanityCheckDeps()

    # ----------------------- public API -----------------------

    def fitPredict(self, embeddings):
        x = self.prepareData(embeddings)
        n = int(x.shape[0])

        if n == 0:
            empty = HdbscanPlusResult(
                labels=np.array([], dtype=np.int32),
                probabilities=np.array([], dtype=np.float32),
                bestScore=-1.0,
                bestParams={},
                scoreDetails={},
                clusterStats={},
                tried=[],
            )
            self.bestResult = empty
            return empty

        if n < 5:
            params = self._defaultParamsForSmallN(n)
            trial = self.evaluateTrial(x, params)
            result = self.makeResultFromTrial(trial, tried=[trial])
            self.bestResult = result
            return result

        best, tried = self.searchBestParams(x)
        if best is None:
            best = tried[-1] if tried else self.evaluateTrial(x, self._defaultParamsForSmallN(n))
            tried = tried or [best]

        result = self.makeResultFromTrial(best, tried=tried)
        self.bestResult = result
        return result

    # ----------------------- deps / sanity -----------------------

    def _sanityCheckDeps(self):
        if hdbscan is None or dbcvValidityIndex is None:
            raise ImportError("hdbscan is required for HDBSCANplus. Install: pip install hdbscan")

    # ----------------------- data prep -----------------------

    def prepareData(self, embeddings):
        x = np.asarray(embeddings, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D embeddings matrix, got shape {x.shape}")

        if self.normalizeVectors:
            x = self._l2Normalize(x)

        # Use euclidean for normalized embeddings.
        self.metricForRun = "euclidean"

        return x

    def _l2Normalize(self, x):
        if skNormalize is not None:
            return skNormalize(x, norm="l2")
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / norms

    # ----------------------- search -----------------------

    def searchBestParams(self, x):
        n = int(x.shape[0])
        ranges = self._effectiveRanges(n)

        base = self._initialParamsForN(n, ranges)
        candidates = deque(self._initialFrontier(base, ranges))
        tried = {}
        expanded = set()

        while candidates and len(tried) < self.maxTrials:
            params = candidates.popleft()
            key = self._paramsKey(params)
            if key in tried:
                continue

            trial = self.evaluateTrial(x, params)
            tried[key] = trial

            if len(tried) >= self.expandTopK or not candidates:
                self._expandTopK(tried, expanded, candidates, ranges)

        if not tried:
            return None, []

        best = min(tried.values(), key=self._rankSortKey)
        return best, list(tried.values())

    def _expandTopK(self, tried, expanded, candidates, ranges):
        ranked = sorted(tried.items(), key=lambda kv: self._rankSortKey(kv[1]))
        expanded_count = 0

        for key, trial in ranked:
            if key in expanded:
                continue

            expanded.add(key)
            expanded_count += 1

            neighbors = self._neighborParams(trial["params"], ranges)
            for nb in neighbors:
                nb_key = self._paramsKey(nb)
                if nb_key not in tried:
                    candidates.append(nb)

            if expanded_count >= self.expandTopK:
                break

    def _rankSortKey(self, trial):
        params = trial["params"]
        return (-trial["score"], int(params["minClusterSize"]), float(params["clusterSelectionEpsilon"]))

    def _effectiveRanges(self, n):
        mcs_min, mcs_max = self.minClusterSizeRange
        mcs_max = min(mcs_max, n)
        mcs_min = min(mcs_min, mcs_max)
        mcs_min = max(2, mcs_min)

        eps_min, eps_max = self.clusterSelectionEpsilonRange
        eps_min = max(0.0, eps_min)
        eps_max = max(eps_min, eps_max)

        return (mcs_min, mcs_max), (eps_min, eps_max)

    def _initialParamsForN(self, n, ranges):
        (mcs_min, mcs_max), (eps_min, eps_max) = ranges
        base = int(max(2, round(math.sqrt(n))))
        mcs = self._clampInt(base, (mcs_min, mcs_max))
        eps = self._clampFloat(0.0, (eps_min, eps_max))
        return {"minClusterSize": mcs, "clusterSelectionEpsilon": eps}

    def _initialFrontier(self, base, ranges):
        (mcs_min, mcs_max), (eps_min, eps_max) = ranges
        mcs = int(base["minClusterSize"])
        eps = float(base["clusterSelectionEpsilon"])

        mcs_candidates = {
            self._clampInt(mcs, (mcs_min, mcs_max)),
            self._clampInt(max(2, int(round(mcs / 2))), (mcs_min, mcs_max)),
            self._clampInt(mcs * 2, (mcs_min, mcs_max)),
        }

        frontier = []
        for mcs_val in sorted(mcs_candidates):
            params = {
                "minClusterSize": int(mcs_val),
                "clusterSelectionEpsilon": float(self._clampFloat(eps, (eps_min, eps_max))),
            }
            frontier.append(params)
        return frontier

    def _neighborParams(self, params, ranges):
        (mcs_min, mcs_max), (eps_min, eps_max) = ranges
        mcs = int(params["minClusterSize"])
        eps = float(params["clusterSelectionEpsilon"])

        neighbors = []

        step_vals = [0.7, 0.9, 1.1, 1.3]
        m_candidates = []
        for step in step_vals:
            m_val = int(round(mcs * step))
            m_val = self._clampInt(m_val, (mcs_min, mcs_max))
            if m_val != mcs:
                m_candidates.append(m_val)

        for m_val in sorted(set(m_candidates)):
            neighbors.append({
                "minClusterSize": int(m_val),
                "clusterSelectionEpsilon": float(self._clampFloat(eps, (eps_min, eps_max))),
            })

        if eps_max > eps_min and self.epsilonStep > 0.0:
            e_candidates = [
                self._clampFloat(eps - self.epsilonStep, (eps_min, eps_max)),
                self._clampFloat(eps + self.epsilonStep, (eps_min, eps_max)),
            ]
            for e_val in sorted(set(e_candidates)):
                if e_val == eps:
                    continue
                neighbors.append({
                    "minClusterSize": int(mcs),
                    "clusterSelectionEpsilon": float(e_val),
                })

        return neighbors

    def _paramsKey(self, params):
        eps = float(params["clusterSelectionEpsilon"])
        if self.epsilonStep > 0:
            eps = round(eps / self.epsilonStep) * self.epsilonStep
        return (int(params["minClusterSize"]), round(eps, 6))

    def _clampParams(self, params, ranges):
        (mcs_min, mcs_max), (eps_min, eps_max) = ranges
        mcs = self._clampInt(params["minClusterSize"], (mcs_min, mcs_max))
        eps = self._clampFloat(params["clusterSelectionEpsilon"], (eps_min, eps_max))
        return {"minClusterSize": int(mcs), "clusterSelectionEpsilon": float(eps)}

    def _clampInt(self, val, rng):
        return int(max(rng[0], min(rng[1], int(val))))

    def _clampFloat(self, val, rng):
        return float(max(rng[0], min(rng[1], float(val))))

    def _defaultParamsForSmallN(self, n):
        mcs = max(2, min(5, int(n)))
        return {"minClusterSize": mcs, "clusterSelectionEpsilon": 0.0}

    # ----------------------- trial evaluation -----------------------

    def evaluateTrial(self, x, params):
        n = int(x.shape[0])
        try:
            labels, probs = self._runHdbscan(x, params)
        except Exception as exc:
            labels = np.full(n, -1, dtype=np.int32)
            probs = np.zeros(n, dtype=np.float32)
            stats = self._clusterStats(labels)
            stats["dbcvRaw"] = -1.0
            return {
                "params": dict(params),
                "score": -1.0,
                "baseScore": -1.0,
                "dbcvRaw": -1.0,
                "dbcvNorm": 0.0,
                "bicRaw": -1.0,
                "bicNull": -1.0,
                "bicScore": 0.0,
                "sanityPenalty": 1.0,
                "mixedPenalty": 0.0,
                "mixedLambda": 0.0,
                "penaltyDetails": {},
                "mixedDetails": {},
                "gate": "error",
                "alpha": self.alpha,
                "stats": stats,
                "labels": labels,
                "probabilities": probs,
                "error": str(exc),
            }

        stats = self._clusterStats(labels)
        dbcvRaw = float(self.safeDbcv(x, labels))
        dbcvNorm = self._normalizeDbcv(dbcvRaw)
        bicScore, bicRaw, bicNull = self._bicScore(x, labels)

        baseScore, gate, alpha = self._baseScore(dbcvNorm, bicScore)
        sanityPenalty, penaltyDetails = self._penalty(stats)
        mixedPenalty, mixedDetails = self._mixedPenalty(x, labels, params["minClusterSize"])
        mixedLambda = self.lambdaMax * dbcvNorm

        score = baseScore - sanityPenalty - (mixedLambda * mixedPenalty)

        return {
            "params": dict(params),
            "score": float(score),
            "baseScore": float(baseScore),
            "dbcvRaw": float(dbcvRaw),
            "dbcvNorm": float(dbcvNorm),
            "bicRaw": float(bicRaw),
            "bicNull": float(bicNull),
            "bicScore": float(bicScore),
            "sanityPenalty": float(sanityPenalty),
            "mixedPenalty": float(mixedPenalty),
            "mixedLambda": float(mixedLambda),
            "penaltyDetails": penaltyDetails,
            "mixedDetails": mixedDetails,
            "gate": gate,
            "alpha": float(alpha),
            "stats": stats,
            "labels": labels,
            "probabilities": probs,
        }

    def _runHdbscan(self, x, params):
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=int(params["minClusterSize"]),
            min_samples=None,
            metric=self.metricForRun,
            cluster_selection_epsilon=float(params.get("clusterSelectionEpsilon", 0.0)),
            cluster_selection_method=self.clusterSelectionMethod,
            prediction_data=False,
        )
        labels = clusterer.fit_predict(x)
        probabilities = getattr(clusterer, "probabilities_", None)
        if probabilities is None:
            probabilities = np.ones_like(labels, dtype=np.float32)
        return np.asarray(labels, dtype=np.int32), np.asarray(probabilities, dtype=np.float32)

    # ----------------------- stats / objectives -----------------------

    def _clusterStats(self, labels):
        labels = np.asarray(labels)
        n = int(labels.shape[0])
        noiseMask = labels == -1
        noiseRate = float(np.mean(noiseMask)) if n else 1.0
        coverage = float(1.0 - noiseRate)

        nonNoiseLabels = [l for l in sorted(set(labels.tolist())) if l != -1]
        clusterSizes = [int((labels == l).sum()) for l in nonNoiseLabels]
        clusterCount = int(len(clusterSizes))

        largestClusterFrac = 0.0
        nonNoiseCount = max(1, int(n - int(noiseMask.sum())))
        if clusterSizes:
            largestClusterFrac = float(max(clusterSizes) / nonNoiseCount)

        tinyCount = 0
        for s in clusterSizes:
            if s < self.minUsefulClusterSize:
                tinyCount += 1
        tinyClusterFrac = float(tinyCount / max(1, clusterCount))

        return {
            "n": n,
            "clusterCount": clusterCount,
            "clusterSizes": clusterSizes,
            "noiseRate": noiseRate,
            "coverage": coverage,
            "largestClusterFrac": largestClusterFrac,
            "tinyClusterCount": tinyCount,
            "tinyClusterFrac": tinyClusterFrac,
        }

    def safeDbcv(self, x, labels):
        try:
            labs = set(labels.tolist())
            nonNoise = [l for l in labs if l != -1]
            if len(nonNoise) < 2:
                return -1.0
            return float(dbcvValidityIndex(x, labels))
        except Exception:
            return -1.0

    def _normalizeDbcv(self, dbcvRaw):
        return float(max(0.0, min(1.0, (float(dbcvRaw) + 1.0) / 2.0)))

    def _bicScore(self, x, labels):
        bic_pair = self._bicForLabels(x, labels)
        if bic_pair is None:
            return 0.0, -1.0, -1.0

        bicRaw, bicNull = bic_pair
        denom = abs(bicNull) + 1e-9
        score = (bicNull - bicRaw) / denom
        score = float(max(0.0, min(1.0, score)))
        return score, float(bicRaw), float(bicNull)

    def _bicForLabels(self, x, labels):
        labels = np.asarray(labels)
        mask = labels != -1
        if int(mask.sum()) == 0:
            return None

        x_use = np.asarray(x[mask], dtype=np.float64)
        labels_use = labels[mask]
        labs = [l for l in sorted(set(labels_use.tolist())) if l != -1]
        if not labs:
            return None

        total_logL = 0.0
        for lab in labs:
            pts = x_use[labels_use == lab]
            total_logL += self._logLikelihoodDiag(pts)

        n = int(x_use.shape[0])
        d = int(x_use.shape[1])
        k = int(len(labs))
        p = k * (2 * d)
        bicRaw = -2.0 * total_logL + p * math.log(max(1, n))

        null_logL = self._logLikelihoodDiag(x_use)
        p_null = 2 * d
        bicNull = -2.0 * null_logL + p_null * math.log(max(1, n))

        return bicRaw, bicNull

    def _logLikelihoodDiag(self, pts):
        if pts.size == 0:
            return 0.0
        pts = np.asarray(pts, dtype=np.float64)
        n, d = pts.shape
        if n == 0:
            return 0.0
        mean = pts.mean(axis=0)
        var = pts.var(axis=0) + self.bicVarEps
        log_det = float(np.sum(np.log(var)))
        const = -0.5 * (d * math.log(2.0 * math.pi) + log_det)
        diffs = pts - mean
        quad = np.sum((diffs * diffs) / var, axis=1)
        return float(np.sum(const - 0.5 * quad))

    def _penalty(self, stats):
        noiseRate = float(stats.get("noiseRate", 1.0))
        tinyFrac = float(stats.get("tinyClusterFrac", 1.0))
        clusterCount = int(stats.get("clusterCount", 0))

        penalty = 0.0
        penalty += self.noisePenaltyWeight * noiseRate
        if clusterCount <= 1:
            penalty += self.singleClusterPenalty
        penalty += self.tinyClusterPenaltyWeight * tinyFrac

        return penalty, {
            "noise": self.noisePenaltyWeight * noiseRate,
            "singleCluster": self.singleClusterPenalty if clusterCount <= 1 else 0.0,
            "tinyClusters": self.tinyClusterPenaltyWeight * tinyFrac,
        }

    def _baseScore(self, dbcvNorm, bicScore):
        if dbcvNorm < self.dbcvGate:
            return dbcvNorm, "dbcv_only", 1.0

        score = self.alpha * dbcvNorm + (1.0 - self.alpha) * bicScore
        return score, "blend", self.alpha

    def _mixedPenalty(self, x, labels, minClusterSize):
        labels = np.asarray(labels)
        n = int(labels.shape[0])
        if n == 0:
            return 0.0, {"skipped": True, "reason": "empty"}

        real_labels = [l for l in sorted(set(labels.tolist())) if l != -1]
        if len(real_labels) < 2:
            return 0.0, {"skipped": True, "reason": "too_few_clusters"}

        m = max(1, int(minClusterSize))
        total = 0.0

        for lab in real_labels:
            idx = np.where(labels == lab)[0]
            if idx.size < 2:
                continue

            pts = x[idx]
            spread_before = self._clusterSpread(pts)
            if spread_before <= 0.0:
                continue

            split_labels = self._kmeans2Labels(pts)
            pts0 = pts[split_labels == 0]
            pts1 = pts[split_labels == 1]
            if pts0.size == 0 or pts1.size == 0:
                continue

            spread0 = self._clusterSpread(pts0)
            spread1 = self._clusterSpread(pts1)
            spread_after = (pts0.shape[0] * spread0 + pts1.shape[0] * spread1) / float(idx.size)

            improvement = (spread_before - spread_after) / spread_before
            improvement = max(0.0, min(1.0, float(improvement)))

            r = float(idx.size) / float(m)
            size_weight = r / (1.0 + r)

            mixed_penalty_cluster = improvement * size_weight
            total += idx.size * mixed_penalty_cluster

        total_penalty = total / float(n)
        return float(total_penalty), {"skipped": False}

    def _clusterSpread(self, pts):
        if pts.size == 0:
            return 0.0
        if pts.shape[0] <= 1:
            return 0.0
        centroid = pts.mean(axis=0)
        dists = np.linalg.norm(pts - centroid, axis=1)
        return float(np.mean(dists))

    def _kmeans2Labels(self, pts):
        n = int(pts.shape[0])
        if n <= 1:
            return np.zeros(n, dtype=np.int32)

        c0 = pts[0]
        dists = np.linalg.norm(pts - c0, axis=1)
        idx1 = int(np.argmax(dists))
        c1 = pts[idx1]

        labels = None
        for _ in range(self.kmeansMaxIter):
            d0 = np.sum((pts - c0) ** 2, axis=1)
            d1 = np.sum((pts - c1) ** 2, axis=1)
            new_labels = (d1 < d0).astype(np.int32)

            if labels is not None and np.array_equal(new_labels, labels):
                break
            labels = new_labels

            if np.all(labels == 0) or np.all(labels == 1):
                break

            c0 = pts[labels == 0].mean(axis=0)
            c1 = pts[labels == 1].mean(axis=0)

        if labels is None:
            labels = np.zeros(n, dtype=np.int32)
        return labels

    # ----------------------- result packaging -----------------------

    def makeResultFromTrial(self, best, tried):
        labels = best.get("labels")
        probabilities = best.get("probabilities")
        if labels is None:
            labels = np.array([], dtype=np.int32)
        if probabilities is None:
            probabilities = np.ones_like(labels, dtype=np.float32)

        stats = best.get("stats", {})
        scoreDetails = {
            "score": float(best.get("score", -1.0)),
            "baseScore": float(best.get("baseScore", -1.0)),
            "dbcvRaw": float(best.get("dbcvRaw", -1.0)),
            "dbcvNorm": float(best.get("dbcvNorm", 0.0)),
            "bicRaw": float(best.get("bicRaw", -1.0)),
            "bicNull": float(best.get("bicNull", -1.0)),
            "bicScore": float(best.get("bicScore", 0.0)),
            "sanityPenalty": float(best.get("sanityPenalty", 0.0)),
            "mixedPenalty": float(best.get("mixedPenalty", 0.0)),
            "mixedLambda": float(best.get("mixedLambda", 0.0)),
            "penaltyDetails": best.get("penaltyDetails", {}),
            "mixedDetails": best.get("mixedDetails", {}),
            "gate": best.get("gate", ""),
            "alpha": float(best.get("alpha", self.alpha)),
            "error": best.get("error", ""),
        }

        clusterStats = dict(stats)
        bestParams = dict(best.get("params", {}))
        bestParams["minSamples"] = "default"
        bestScore = float(best.get("score", -1.0))

        return HdbscanPlusResult(
            labels=np.asarray(labels),
            probabilities=np.asarray(probabilities),
            bestScore=bestScore,
            bestParams=bestParams,
            scoreDetails=scoreDetails,
            clusterStats=clusterStats,
            tried=tried,
        )


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    a = rng.normal(size=(200, 32)) * 0.4 + 0.0
    b = rng.normal(size=(200, 32)) * 0.4 + 3.0
    x = np.vstack([a, b]).astype(np.float32)

    clusterer = HDBSCANplus(
        metric="cosine",
        normalizeVectors=True,
        maxTrials=60,
        debug=True,
    )
    result = clusterer.fitPredict(x)
    print("Best score:", result.bestScore)
    print("Best params:", result.bestParams)
    print("Score details:", result.scoreDetails)
    print("Cluster stats:", result.clusterStats)
    print("Unique labels:", sorted(set(result.labels.tolist())))
