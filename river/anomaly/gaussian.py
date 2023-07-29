from __future__ import annotations

from river import anomaly, proba, utils


class GaussianScorer(anomaly.base.SupervisedAnomalyDetector):
    """Univariate Gaussian anomaly detector.

    This is a supervised anomaly detector. It fits a Gaussian distribution to the target values.
    The anomaly score is then computed as so:

    $$score = 2 \\mid CDF(y) - 0.5 \\mid$$

    This makes it so that the anomaly score is between 0 and 1.

    Parameters
    ----------
    window_size
        Set this to fit the Gaussian distribution over a window of recent values.
    grace_period
        Number of samples before which a 0 is always returned. This is handy because the Gaussian
        distribution needs time to stabilize, and will likely produce overly high anomaly score
        for the first samples.

    Examples
    --------

    >>> import random
    >>> from river import anomaly

    >>> rng = random.Random(42)
    >>> detector = anomaly.GaussianScorer()

    >>> for y in (rng.gauss(0, 1) for _ in range(100)):
    ...     detector = detector.learn_one(None, y)

    >>> detector.score_one(None, -3)
    0.999477...

    >>> detector.score_one(None, 3)
    0.999153...

    >>> detector.score_one(None, 0)
    0.052665...

    >>> detector.score_one(None, 0.5)
    0.383717...

    """

    def __init__(self, window_size=None, grace_period=100):
        self.window_size = window_size
        self.gaussian = (
            utils.Rolling(proba.Gaussian(), window_size=self.window_size)
            if window_size
            else proba.Gaussian()
        )
        self.grace_period = grace_period

    def learn_one(self, x, y):
        self.gaussian.update(y)
        return self

    def score_one(self, x, y):
        if self.gaussian.n_samples < self.grace_period:
            return 0
        return 2 * abs(self.gaussian.cdf(y) - 0.5)
