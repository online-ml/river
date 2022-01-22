import math

from river.stats.var import RollingVar

from .base import AnomalyDetector


class Zscore(AnomalyDetector):
    """
    Anomaly detection based on zscore. Data is assumed to have gaussian distribution

    Examples
    --------
    import numppy as np
    from river.anomaly.zscore import Zscore

    #warmup anomaly detector
    zs = Zscore(100)
    mean = 10.0
    sd = 2.0

    for _ in range(110):
        v = np.random.normal(mean, sd)
        zs.learn_one(v)

    #get score
    v = mean + 4 * sd
    sc = zs.score_one(v)
    print(sc)

    """

    def __init__(self, window_size: int, zs_threshold: float = 3.0):
        """

        Parameters
        ----------
        window_size : size of rolling window
        zs_threshold : zscore threshold
        """
        self.var = RollingVar(window_size)
        self.zs_threshold = zs_threshold

    def learn_one(self, x: float):
        """
        Parameters
        ----------
        x the value
        Returns
        -------
        """
        self.var.update(x)

    def score_one(self, x: float) -> float:
        """
        Parameters
        ----------
        x   the value
        Returns
        -------
        anomaly score between 0 and 1
        """
        sd = math.sqrt(self.var.get())
        mean = self.var._rolling_mean.get()
        zs = abs(x - mean) / sd
        score = 1.0 if zs > self.zs_threshold else zs / self.zs_threshold
        return score
