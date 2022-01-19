import math
from base import AnomalyDetector
from river.stats.var import RollingVar


class Zscore(AnomalyDetector):
    """ """

    def __init__(self, window_size: int, zs_threshold: float = 3.0):
        """

        Parameters
        ----------
        window_size
        zs_threshold
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
