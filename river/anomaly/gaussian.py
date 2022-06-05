from river import anomaly, proba
import datetime as dt
import pytz

class GaussianScorer(anomaly.base.SupervisedAnomalyDetector):

    """Univariate Gaussian anomaly detector.

    This is a supervised anomaly detector. It fits a Gaussian distribution to the target values.
    The anomaly score is then computed as so:

    $$score = 2 * \\mid CDF(y) - 0.5 \\mid$$

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

    def __init__(self, window_size=None, window_size_t=None, timestamp_label=None, timestamp_format='%Y-%m-%dT%H:%M:%S%z', grace_period=100):
        self.window_size = window_size
        self.window_size_t = window_size_t
        self.timestamp_label=timestamp_label
        self.timestamp_format=timestamp_format
        self.grace_period_passed = False
        self.grace_period_counter = 0
        self.gaussian = (
            proba.Rolling(proba.Gaussian(), window_size=self.window_size) if window_size 
            else proba.TimeRolling(proba.Gaussian(), period=self.window_size_t) if window_size_t 
            else proba.Gaussian()
        )
        self.grace_period = grace_period

    @property
    def _dist(self):
        return (
            self.gaussian
            if isinstance(self.gaussian, proba.Gaussian)
            else self.gaussian.dist
        )

    def learn_one(self, _, y):
        if not self.grace_period_passed and self.grace_period:
            self.grace_period_counter += 1
            if self.grace_period_counter > self.grace_period: 
                self.grace_period_passed = True
        if isinstance(self.gaussian, proba.TimeRolling):
            t = _[self.timestamp_label] if isinstance(_, dict) else _
            parsed_t = (
                t if isinstance(t,dt.datetime)
                else dt.datetime.utcfromtimestamp(t)  if isinstance(t,int)
                else dt.datetime.strptime(t,self.timestamp_format)
            )
            parsed_t =  parsed_t if parsed_t.tzinfo else pytz.utc.localize(parsed_t)
            self.gaussian.update(y, parsed_t)
        else:
            self.gaussian.update(y)
        return self

    def score_one(self, _, y):
        if self._dist.n_samples < self.grace_period and not self.grace_period_passed:
            return 0
        return 2 * abs(self._dist.cdf(y) - 0.5)
