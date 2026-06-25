from __future__ import annotations

import math
import random

from river import anomaly, sketch

__all__ = ["LODA"]


class LODA(anomaly.base.AnomalyDetector):
    """LODA (Lightweight on-line detector of anomalies).

    LODA [^1] is an ensemble of one-dimensional histograms. Each histogram approximates the
    probability density of the data once it has been projected onto a sparse random vector. The
    anomaly score of a sample is the average negative log-likelihood of its projections across the
    ensemble: rare projected values yield low densities and therefore high scores.

    Pevný showed that aggregating many such deliberately weak detectors yields a strong anomaly
    detector, competitive with much heavier methods while remaining cheap to update online.

    Each projection vector is sparse: only `⌊√d⌋` of the `d` features have a non-zero weight, drawn
    from a standard normal distribution. The feature set and the projections are fixed the first
    time `learn_one` is called. Features that appear later are ignored, and missing features are
    treated as zeros.

    Unlike the histograms used in the original paper, this implementation relies on River's
    streaming `sketch.Histogram`, which maintains a bounded number of adaptive-width bins. The
    density of a projected value is estimated as `(count / n) / width` of the bin that contains it,
    where `width` is the bin's span (or, for not-yet-merged singleton bins, the distance to the
    nearest neighbouring bin). Projected values that fall outside every bin are assigned a floor
    density, making them maximally anomalous. This keeps the detector fully online and free of any
    numpy dependency.

    Parameters
    ----------
    n_bins
        Maximum number of bins in each histogram.
    n_random_cuts
        Number of random projections (the ensemble size).
    seed
        Random number seed, for reproducible projections.

    Attributes
    ----------
    n_features
        Number of features seen during the first call to `learn_one`.

    Examples
    --------

    >>> from river import anomaly
    >>> from river import datasets

    >>> loda = anomaly.LODA(n_bins=10, n_random_cuts=100, seed=42)

    >>> for x, y in datasets.CreditCard().take(2500):
    ...     loda.learn_one(x)

    >>> loda.n_features
    30

    >>> score = loda.score_one(x)
    >>> print(f"{score:.3f}")
    3.670

    References
    ----------
    [^1]: Pevný, T., 2016. Loda: Lightweight on-line detector of anomalies. Machine Learning,
    102(2), pp.275-304.

    """

    def __init__(self, n_bins: int = 10, n_random_cuts: int = 100, seed: int | None = None):
        self.n_bins = n_bins
        self.n_random_cuts = n_random_cuts
        self.seed = seed
        self.rng = random.Random(seed)
        self.features_: list = []
        self.projections_: list[dict] = []
        self.histograms_: list[sketch.Histogram] = []

    @property
    def n_features(self) -> int:
        return len(self.features_)

    def _initialize(self, x: dict) -> None:
        """Fix the feature set and draw the sparse random projections."""
        self.features_ = sorted(x.keys())
        d = len(self.features_)
        n_nonzero = max(1, int(math.sqrt(d)))
        for _ in range(self.n_random_cuts):
            chosen = self.rng.sample(self.features_, n_nonzero)
            self.projections_.append({feature: self.rng.gauss(0.0, 1.0) for feature in chosen})
            self.histograms_.append(sketch.Histogram(max_bins=self.n_bins))

    @staticmethod
    def _project(projection: dict, x: dict) -> float:
        return sum(weight * x.get(feature, 0.0) for feature, weight in projection.items())

    def learn_one(self, x):
        if not self.projections_:
            self._initialize(x)
        for projection, histogram in zip(self.projections_, self.histograms_):
            histogram.update(self._project(projection, x))

    @staticmethod
    def _density(histogram: sketch.Histogram, z: float) -> float:
        """Probability density of ``z`` under a one-dimensional streaming histogram."""
        n = histogram.n
        if n == 0:
            return 0.0
        bins = histogram.data
        for i, b in enumerate(bins):
            if z > b.right:
                continue
            if z < b.left:
                # z falls in a gap before this bin: no probability mass there.
                return 0.0
            width = b.right - b.left
            if width > 0:
                return (b.count / n) / width
            # Singleton bin (left == right): borrow a width from the nearest neighbour.
            left_gap = b.left - bins[i - 1].right if i > 0 else math.inf
            right_gap = bins[i + 1].left - b.right if i + 1 < len(bins) else math.inf
            width = min(left_gap, right_gap)
            if not math.isfinite(width) or width <= 0:
                # Degenerate histogram with a single point: fall back to probability mass.
                return b.count / n
            return (b.count / n) / width
        # z is to the right of every bin: no probability mass there.
        return 0.0

    def score_one(self, x):
        if not self.projections_:
            return 0.0
        score = 0.0
        for projection, histogram in zip(self.projections_, self.histograms_):
            z = self._project(projection, x)
            p = self._density(histogram, z)
            score -= math.log(max(p, 1e-12))
        return score / self.n_random_cuts

    @classmethod
    def _unit_test_params(cls):
        yield {"n_random_cuts": 10, "seed": 42}
