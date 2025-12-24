# Self-converging HDBSCAN wrapper (HDBSCAN+)
# Pipeline:
# 1) Embeddings (optional L2-normalized)
# 2) HDBSCAN (parameterized)
# 3) Score = DBCV floor + (optional) BIC ceiling - penalties
# 4) Local neighbor expansion search until no improvements

from __future__ import annotations

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
        minSamplesRange=(1, 80),
        clusterSelectionEpsilonRange=(0.0, 0.0),
        neighborStepFrac=0.2,
        epsilonStep=0.05,
        dbcvGate=0.4,
        alpha=0.7,
        noisePenaltyWeight=0.6,
        singleClusterPenalty=0.4,
        tinyClusterPenaltyWeight=0.3,
        minUsefulClusterSize=4,
        bicVarEps=1e-6,
    ):
        self.metric = metric
        self.normalizeVectors = bool(normalizeVectors)
        self.cosineViaNormalize = bool(cosineViaNormalize)
        self.clusterSelectionMethod = clusterSelectionMethod
        self.maxTrials = int(max(1, maxTrials))
        self.randomState = int(randomState)
        self.debug = bool(debug)

        self.minClusterSizeRange = (int(minClusterSizeRange[0]), int(minClusterSizeRange[1]))
        self.minSamplesRange = (int(minSamplesRange[0]), int(minSamplesRange[1]))
        self.clusterSelectionEpsilonRange = (
            float(clusterSelectionEpsilonRange[0]),
            float(clusterSelectionEpsilonRange[1]),
        )

        self.neighborStepFrac = float(max(0.05, neighborStepFrac))
        self.epsilonStep = float(max(0.0, epsilonStep))

        self.dbcvGate = float(max(0.0, min(1.0, dbcvGate)))
        self.alpha = float(max(0.0, min(1.0, alpha)))

        self.noisePenaltyWeight = float(max(0.0, noisePenaltyWeight))
        self.singleClusterPenalty = float(max(0.0, singleClusterPenalty))
        self.tinyClusterPenaltyWeight = float(max(0.0, tinyClusterPenaltyWeight))
        self.minUsefulClusterSize = int(max(2, minUsefulClusterSize))
        self.bicVarEps = float(max(1e-12, bicVarEps))

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

        if self.metric == "cosine" and self.cosineViaNormalize:
            self.metricForRun = "euclidean"
        else:
            self.metricForRun = self.metric

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
        frontier = self._initialFrontier(base, ranges)
        visited = set()
        tried = []
        best = None

        while frontier and len(tried) < self.maxTrials:
            params = frontier.pop(0)
            params = self._clampParams(params, ranges)
            key = self._paramsKey(params)
            if key in visited:
                continue
            visited.add(key)

            trial = self.evaluateTrial(x, params)
            tried.append(trial)

            if best is None or trial["score"] > best["score"]:
                best = trial
                neighbors = self._neighborParams(params, ranges)
                for nb in neighbors:
                    nb_key = self._paramsKey(nb)
                    if nb_key not in visited:
                        frontier.append(nb)

        return best, tried

    def _effectiveRanges(self, n):
        mcs_min, mcs_max = self.minClusterSizeRange
        mcs_max = min(mcs_max, n)
        mcs_min = min(mcs_min, mcs_max)

        ms_min, ms_max = self.minSamplesRange
        ms_max = min(ms_max, n)
        ms_min = min(ms_min, ms_max)

        eps_min, eps_max = self.clusterSelectionEpsilonRange
        eps_min = max(0.0, eps_min)
        eps_max = max(eps_min, eps_max)

        return (mcs_min, mcs_max), (ms_min, ms_max), (eps_min, eps_max)

    def _initialParamsForN(self, n, ranges):
        (mcs_min, mcs_max), (ms_min, ms_max), (eps_min, eps_max) = ranges
        base = int(max(2, round(math.sqrt(n))))
        mcs = self._clampInt(base, (mcs_min, mcs_max))

        ms = int(max(1, round(mcs / 2)))
        ms = self._clampInt(ms, (ms_min, ms_max))
        ms = min(ms, mcs)

        eps = self._clampFloat(0.0, (eps_min, eps_max))
        return {"minClusterSize": mcs, "minSamples": ms, "clusterSelectionEpsilon": eps}

    def _initialFrontier(self, base, ranges):
        (mcs_min, mcs_max), (ms_min, ms_max), (eps_min, eps_max) = ranges
        mcs = int(base["minClusterSize"])
        eps = float(base["clusterSelectionEpsilon"])

        mcs_candidates = {
            self._clampInt(mcs, (mcs_min, mcs_max)),
            self._clampInt(max(2, mcs // 2), (mcs_min, mcs_max)),
            self._clampInt(mcs * 2, (mcs_min, mcs_max)),
        }

        frontier = []
        for mcs_val in sorted(mcs_candidates):
            ms_candidates = {1, max(1, mcs_val // 2), mcs_val}
            for ms_val in sorted(ms_candidates):
                ms_val = self._clampInt(ms_val, (ms_min, ms_max))
                ms_val = min(ms_val, mcs_val)
                params = {
                    "minClusterSize": int(mcs_val),
                    "minSamples": int(ms_val),
                    "clusterSelectionEpsilon": float(self._clampFloat(eps, (eps_min, eps_max))),
                }
                frontier.append(params)
        return frontier

    def _neighborParams(self, params, ranges):
        (mcs_min, mcs_max), (ms_min, ms_max), (eps_min, eps_max) = ranges

        mcs = int(params["minClusterSize"])
        ms = int(params["minSamples"])
        eps = float(params["clusterSelectionEpsilon"])

        step_mcs = max(1, int(round(mcs * self.neighborStepFrac)))
        step_ms = max(1, int(round(max(1, ms) * self.neighborStepFrac)))

        neighbors = []

        for delta in (-step_mcs, step_mcs):
            m_val = self._clampInt(mcs + delta, (mcs_min, mcs_max))
            ms_val = min(ms, m_val)
            ms_val = self._clampInt(ms_val, (ms_min, ms_max))
            neighbors.append({
                "minClusterSize": int(m_val),
                "minSamples": int(ms_val),
                "clusterSelectionEpsilon": float(self._clampFloat(eps, (eps_min, eps_max))),
            })

        for delta in (-step_ms, step_ms):
            ms_val = self._clampInt(ms + delta, (ms_min, ms_max))
            ms_val = min(ms_val, mcs)
            neighbors.append({
                "minClusterSize": int(mcs),
                "minSamples": int(ms_val),
                "clusterSelectionEpsilon": float(self._clampFloat(eps, (eps_min, eps_max))),
            })

        if eps_max > eps_min and self.epsilonStep > 0.0:
            for delta in (-self.epsilonStep, self.epsilonStep):
                e_val = self._clampFloat(eps + delta, (eps_min, eps_max))
                neighbors.append({
                    "minClusterSize": int(mcs),
                    "minSamples": int(ms),
                    "clusterSelectionEpsilon": float(e_val),
                })

        return neighbors

    def _paramsKey(self, params):
        eps = float(params["clusterSelectionEpsilon"])
        if self.epsilonStep > 0:
            eps = round(eps / self.epsilonStep) * self.epsilonStep
        return (int(params["minClusterSize"]), int(params["minSamples"]), round(eps, 6))

    def _clampParams(self, params, ranges):
        (mcs_min, mcs_max), (ms_min, ms_max), (eps_min, eps_max) = ranges
        mcs = self._clampInt(params["minClusterSize"], (mcs_min, mcs_max))
        ms = self._clampInt(params["minSamples"], (ms_min, ms_max))
        ms = min(ms, mcs)
        eps = self._clampFloat(params["clusterSelectionEpsilon"], (eps_min, eps_max))
        return {"minClusterSize": int(mcs), "minSamples": int(ms), "clusterSelectionEpsilon": float(eps)}

    def _clampInt(self, val, rng):
        return int(max(rng[0], min(rng[1], int(val))))

    def _clampFloat(self, val, rng):
        return float(max(rng[0], min(rng[1], float(val))))

    def _defaultParamsForSmallN(self, n):
        mcs = max(2, min(5, int(n)))
        ms = max(1, min(mcs, int(n)))
        return {"minClusterSize": mcs, "minSamples": ms, "clusterSelectionEpsilon": 0.0}

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
                "dbcvRaw": -1.0,
                "dbcvNorm": 0.0,
                "bicRaw": -1.0,
                "bicNull": -1.0,
                "bicScore": 0.0,
                "penalty": 1.0,
                "penaltyDetails": {},
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

        penalty, penaltyDetails = self._penalty(stats)
        score, gate, alpha = self._blendScore(dbcvNorm, bicScore, penalty)

        return {
            "params": dict(params),
            "score": float(score),
            "dbcvRaw": float(dbcvRaw),
            "dbcvNorm": float(dbcvNorm),
            "bicRaw": float(bicRaw),
            "bicNull": float(bicNull),
            "bicScore": float(bicScore),
            "penalty": float(penalty),
            "penaltyDetails": penaltyDetails,
            "gate": gate,
            "alpha": float(alpha),
            "stats": stats,
            "labels": labels,
            "probabilities": probs,
        }

    def _runHdbscan(self, x, params):
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=int(params["minClusterSize"]),
            min_samples=int(params["minSamples"]),
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

    def _blendScore(self, dbcvNorm, bicScore, penalty):
        if dbcvNorm < self.dbcvGate:
            return dbcvNorm - penalty, "dbcv_only", 1.0

        score = self.alpha * dbcvNorm + (1.0 - self.alpha) * bicScore - penalty
        return score, "blend", self.alpha

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
            "dbcvRaw": float(best.get("dbcvRaw", -1.0)),
            "dbcvNorm": float(best.get("dbcvNorm", 0.0)),
            "bicRaw": float(best.get("bicRaw", -1.0)),
            "bicNull": float(best.get("bicNull", -1.0)),
            "bicScore": float(best.get("bicScore", 0.0)),
            "penalty": float(best.get("penalty", 0.0)),
            "penaltyDetails": best.get("penaltyDetails", {}),
            "gate": best.get("gate", ""),
            "alpha": float(best.get("alpha", self.alpha)),
            "error": best.get("error", ""),
        }

        clusterStats = dict(stats)
        bestParams = dict(best.get("params", {}))
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
