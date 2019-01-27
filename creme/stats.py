"""
Module for computing running statistics
"""
import abc


__all__ = ['Count', 'Mean', 'SmoothMean', 'Variance']


class RunningStatistic(abc.ABC):

    @property
    @abc.abstractmethod
    def name(self):
        pass

    @abc.abstractmethod
    def update(self):
        pass

    @abc.abstractmethod
    def get(self) -> float:
        pass


class Count(RunningStatistic):

    def __init__(self):
        super().__init__()
        self.n = 0

    @property
    def name(self):
        return 'count'

    def update(self, x=None):
        self.n += 1
        return self

    def get(self):
        return self.n


class Mean(RunningStatistic):

    def __init__(self):
        super().__init__()
        self.count = Count()
        self.mu = 0

    @property
    def name(self):
        return 'mean'

    def update(self, x):
        self.mu += (x - self.mu) / self.count.update().get()
        return self

    def get(self):
        return self.mu


class SmoothMean(Mean):
    """Computes the running mean using additive smoothing.

    - https://www.wikiwand.com/en/Additive_smoothing
    """

    def __init__(self, prior, prior_weight):
        super().__init__()
        self.prior = prior
        self.prior_weight = prior_weight

    @property
    def name(self):
        return 'smooth_mean'

    def get(self):
        numerator = self.mu * self.count.get() + self.prior * self.prior_weight
        denominator = self.count.get() + self.prior_weight
        return numerator / denominator


class Variance(RunningStatistic):

    def __init__(self):
        super().__init__()
        self.mean = Mean()
        self.sos = 0

    @property
    def name(self):
        return 'variance'

    def update(self, x):
        old_mean = self.mean.get()
        new_mean = self.mean.update(x).get()
        self.sos += (x - old_mean) * (x - new_mean)
        return self

    def get(self):
        return self.sos / self.mean.count.n if self.sos else 0
