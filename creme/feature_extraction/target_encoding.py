import collections

from .. import base
from .. import stats


class TargetEncoder(base.Transformer):
    """Computes the conditional mean of the target using additive smoothing.

    The formulation is:

    .. math:: \\mu = \\frac{Cm + \\frac{1}{n} \\sum_{i=1}^n x_i}{C + n}

    where $m$ is the global mean, $C$ is the prior weight assigned to the global mean, $n$ is the
    number of observations, and $x_i$ are the observed values.

    Example:

    ::

        >>> from pprint import pprint
        >>> from creme import compose
        >>> from creme import feature_extraction

        >>> data = [
        ...     {'city': 'Tokyo', 'name': 'Peanut butter', 'price': 2},
        ...     {'city': 'Paris', 'name': 'Peanut butter', 'price': 4},
        ...     {'city': 'Paris', 'name': 'Egg and bacon', 'price': 18},
        ...     {'city': 'Tokyo', 'name': 'Egg and bacon', 'price': 4},
        ...     {'city': 'Tokyo', 'name': 'Moules frites', 'price': 10},
        ...     {'city': 'Paris', 'name': 'Moules frites', 'price': 3},
        ...     {'city': 'Paris', 'name': 'Teriyaki rice', 'price': 3},
        ...     {'city': 'Tokyo', 'name': 'Teriyaki rice', 'price': 8},
        ...     {'city': 'Paris', 'name': 'Craby risotto', 'price': 3},
        ...     {'city': 'Tokyo', 'name': 'Craby risotto', 'price': 6},
        ... ]

        >>> extractor = compose.TransformerUnion([
        ...     ('by_city', feature_extraction.TargetEncoder(by='city', prior_weight=1)),
        ...     ('by_name', feature_extraction.TargetEncoder(by='name', prior_weight=1))
        ... ])
        >>> for x in data:
        ...     y = x.pop('price')
        ...     pprint(extractor.transform_one(x))
        ...     extractor = extractor.fit_one(x, y)
        {'target_mean_by_city': 0.0, 'target_mean_by_name': 0.0}
        {'target_mean_by_city': 2.0, 'target_mean_by_name': 2.0}
        {'target_mean_by_city': 3.5, 'target_mean_by_name': 3.0}
        {'target_mean_by_city': 5.0, 'target_mean_by_name': 13.0}
        {'target_mean_by_city': 4.333333..., 'target_mean_by_name': 7.0}
        {'target_mean_by_city': 9.866666..., 'target_mean_by_name': 8.8}
        {'target_mean_by_city': 7.958333...,
         'target_mean_by_name': 6.833333...}
        {'target_mean_by_city': 5.571428...,
         'target_mean_by_name': 4.642857...}
        {'target_mean_by_city': 6.9, 'target_mean_by_name': 6.5}
        {'target_mean_by_city': 6.022222...,
         'target_mean_by_name': 4.555555...}

        >>> data = [
        ...     {'city': 'Tokyo', 'name': 'Peanut butter', 'price': 2},
        ...     {'city': 'Tokyo', 'name': 'Teriyaki rice', 'price': 4},
        ...     {'city': 'Tokyo', 'name': 'Teriyaki rice', 'price': 18},
        ...     {'city': 'Tokyo', 'name': 'Teriyaki rice', 'price': 4},
        ...     {'city': 'Tokyo', 'name': 'Peanut butter', 'price': 10},
        ...     {'city': 'Paris', 'name': 'Moules frites', 'price': 3},
        ...     {'city': 'Paris', 'name': 'Teriyaki rice', 'price': 3},
        ...     {'city': 'Paris', 'name': 'Teriyaki rice', 'price': 8},
        ...     {'city': 'Paris', 'name': 'Craby risotto', 'price': 3},
        ...     {'city': 'Paris', 'name': 'Craby risotto', 'price': 6},
        ... ]

        >>> extractor = feature_extraction.TargetEncoder(by=['city', 'name'], prior_weight=1)
        >>> for x in data:
        ...     y = x.pop('price')
        ...     pprint(extractor.transform_one(x))
        ...     extractor = extractor.fit_one(x, y)
        {'target_mean_by_city_and_name': 0.0}
        {'target_mean_by_city_and_name': 2.0}
        {'target_mean_by_city_and_name': 3.5}
        {'target_mean_by_city_and_name': 10.0}
        {'target_mean_by_city_and_name': 4.5}
        {'target_mean_by_city_and_name': 7.6}
        {'target_mean_by_city_and_name': 6.833333...}
        {'target_mean_by_city_and_name': 4.642857...}
        {'target_mean_by_city_and_name': 6.5}
        {'target_mean_by_city_and_name': 4.555555...}

    References:

    1. `Additive smoothing <https://www.wikiwand.com/en/Additive_smoothing>`_
    2. `Bayesian average <https://www.wikiwand.com/en/Bayesian_average>`_
    3. `Practical example of Bayes estimators <https://www.wikiwand.com/en/Bayes_estimator#/Practical_example_of_Bayes_estimators>`_

    """

    def __init__(self, by, prior_weight=100):
        self.by = by if isinstance(by, list) else [by]
        self.prior_weight = prior_weight
        self.global_mean = stats.Mean()
        self.feature_means = collections.defaultdict(stats.Mean)

    def make_key(self, x):
        return '_'.join(x[i] for i in self.by)

    def fit_one(self, x, y):
        self.global_mean.update(y)
        self.feature_means[self.make_key(x)].update(y)
        return self

    def transform_one(self, x):

        global_mean = self.global_mean.get()
        mean = self.feature_means[self.make_key(x)]
        mu = mean.get()
        count = mean.n
        smooth_mean = (mu * count + global_mean * self.prior_weight) / (count + self.prior_weight)

        return {f'target_mean_by_{"_and_".join(self.by)}': smooth_mean}

    def is_supervised(self):
        return True
