from .constant import Constant

import creme.stats

__all__ = ['NumericalImputer']


class NumericalImputer:
    """ Imputer allow to replace missing values with descriptive statistics.

    Args:
        on (str): Name of the field to impute.
        constant (float): Constant to replace missing values, when using strategy='constant'.

    Attributes:
        imputer (creme.stats): Object to impute missing values.

    >>> import pprint
    >>> import creme
    >>> import numpy as np
    >>> from sklearn import preprocessing
    >>> np.random.seed(42)

    >>> X = [{'x': v} for v in np.random.normal(loc=0, scale=1, size=5)]
    >>> X = [
    ...     {'x': 1.0},
    ...     {'x': 2.0},
    ...     {'x': 3.0},
    ...     {}
    ... ]

    >>> imputer_mean = creme.imputer.NumericalImputer(on='x', strategy='mean')
    >>> for x in X:
    ...     print(imputer_mean.fit_one(x))
    {'x': 1.0}
    {'x': 2.0}
    {'x': 3.0}
    {'x': 2.0}

    >>> imputer_min = creme.imputer.NumericalImputer(on='x', strategy='min')
    >>> for x in X:
    ...     print(imputer_min.fit_one(x))
    {'x': 1.0}
    {'x': 2.0}
    {'x': 3.0}
    {'x': 1.0}

    >>> imputer_max = creme.imputer.NumericalImputer(on='x', strategy='max')
    >>> for x in X:
    ...     print(imputer_max.fit_one(x))
    {'x': 1.0}
    {'x': 2.0}
    {'x': 3.0}
    {'x': 3.0}

    >>> imputer_constant = creme.imputer.NumericalImputer(
    ...     on='x',
    ...     strategy='constant',
    ...     constant_value=0.0,
    ... )

    >>> for x in X:
    ...     print(imputer_constant.fit_one(x))
    {'x': 1.0}
    {'x': 2.0}
    {'x': 3.0}
    {'x': 0.0}

    """

    def __init__(self, on, strategy, constant_value=None, aggregate=None):

        self.on = on

        allowed_imputers = {
            'mean': creme.stats.Mean(),
            'max': creme.stats.Max(),
            'min': creme.stats.Min(),
            'constant': Constant(constant_value=constant_value),
        }

        if strategy not in allowed_imputers:
            raise ValueError(
                f'Can only use these strategies: {allowed_imputers.keys()}\
                    got strategy = {strategy}'
            )

        self.imputer = allowed_imputers[strategy]

    def fit_one(self, x):
        if self.on in x:
            self.imputer.update(x[self.on])
            return x
        return self.transform_one(x)

    def transform_one(self, x):
        if self.on not in x:
            return {
                **x,
                self.on: self.imputer.get()
            }
        return x
