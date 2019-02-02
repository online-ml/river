from . import base
from . import count


class Mean(base.RunningStatistic):
    """Computes a running mean.

    Attributes:
        count (stats.Count)
        mu (float): The current estimated mean.

    """

    def __init__(self):
        self.count = count.Count()
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
    """Computes a running mean using additive smoothing.

    Parameters:
        prior (float): Prior mean.
        prior_weight (float): Strengh of the prior mean.

    Attributes:
        count (stats.Count)
        mu (float): The current estimated mean.

    References:

    - `Additive smoothing <https://www.wikiwand.com/en/Additive_smoothing>`_

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
