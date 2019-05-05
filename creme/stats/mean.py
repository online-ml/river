from . import base


class Mean(base.Univariate):
    """Running mean.

    Attributes:
        mean (float): The running mean.
        n (int): The current number of observations.

    Example:

        ::

            >>> from creme import stats

            >>> X = [-5, -3, -1, 1, 3, 5]
            >>> mean = stats.Mean()
            >>> for x in X:
            ...     print(mean.update(x).get())
            -5.0
            -4.0
            -3.0
            -2.0
            -1.0
            0.0

    """

    def __init__(self, n=0, mean=0):
        self.n = n
        self.mean = mean

    @property
    def name(self):
        return 'mean'

    def update(self, x):
        self.n += 1
        self.mean += (x - self.mean) / self.n if self.n > 0 else 0
        return self

    def get(self):
        return self.mean

    def __add__(self, other):
        return Mean(n=1, mean=self.mean + other.mean)

    def __mul__(self, other):
        return Mean(n=self.n, mean=self.mean * other)


class BayesianMean(base.Univariate):
    """Estimates a mean using outside information.

    References:

        1. `Additive smoothing <https://www.wikiwand.com/en/Additive_smoothing>`_
        2. `Bayesian average <https://www.wikiwand.com/en/Bayesian_average>`_
        3. `Practical example of Bayes estimators <https://www.wikiwand.com/en/Bayes_estimator#/Practical_example_of_Bayes_estimators>`_

    """

    def __init__(self, prior, prior_weight):
        self.prior = prior
        self.prior_weight = prior_weight
        self.mean = Mean()

    @property
    def name(self):
        return 'bayes_mean'

    def update(self, x):
        self.mean.update(x)
        return self

    def get(self):

        # Uses the notation from https://www.wikiwand.com/en/Bayes_estimator#/Practical_example_of_Bayes_estimators
        R = self.mean.get()
        v = self.mean.n
        m = self.prior_weight
        C = self.prior

        return (R * v + C * m) / (v + m)
