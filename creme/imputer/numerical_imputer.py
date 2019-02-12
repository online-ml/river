from .constant import Constant

import creme.stats

__all__ = ['NumericalImputer']


class NumericalImputer:
    '''
    Imputer allow to replace missing values with descriptive statistics.

    Args:
        missing_values (string): Shape of the missing values.
        strategy (string): Name of the function to replace missing values.
        constant (float): Constant to replace missing values.

    Attributes:
        allowed_strategies (list): List of allowed strategies to impute missing values.
        functions_dictionnary (dic): Mapping between allowed strategies and creme functions.
        impute_function (creme.stats): Function to impute missing values.

    >>> import pprint
    >>> import creme
    >>> import numpy as np
    >>> from sklearn import preprocessing
    >>> np.random.seed(42)

    >>> X = [{'x': v} for v in np.random.normal(loc=0, scale=1, size=5)]
    >>> X.append({'y':10})

    >>> imputer_mean = creme.imputer.NumericalImputer(on='x', strategy='mean')
    >>> pprint.pprint([imputer_mean.fit_one(x).transform_one(x) for x in X])
    [{'x': 0.4967141530112327},
     {'x': -0.13826430117118466},
     {'x': 0.6476885381006925},
     {'x': 1.5230298564080254},
     {'x': -0.23415337472333597},
     {'x': 0.45900297432508597, 'y': 10}]

    >>> imputer_min = creme.imputer.NumericalImputer(on='x', strategy='min')
    >>> pprint.pprint([imputer_min.fit_one(x).transform_one(x) for x in X])
    [{'x': 0.4967141530112327},
     {'x': -0.13826430117118466},
     {'x': 0.6476885381006925},
     {'x': 1.5230298564080254},
     {'x': -0.23415337472333597},
     {'x': -0.23415337472333597, 'y': 10}]

    >>> imputer_max = creme.imputer.NumericalImputer(
    ...    on='x',
    ...    strategy='max',
    ... )

    >>> pprint.pprint([imputer_max.fit_one(x).transform_one(x) for x in X])
    [{'x': 0.4967141530112327},
     {'x': -0.13826430117118466},
     {'x': 0.6476885381006925},
     {'x': 1.5230298564080254},
     {'x': -0.23415337472333597},
     {'x': 1.5230298564080254, 'y': 10}]

    >>> imputer_constant = creme.imputer.NumericalImputer(
    ...     on='x',
    ...     strategy='constant',
    ...     constant_value=0,
    ... )

    >>> pprint.pprint([imputer_constant.fit_one(x).transform_one(x) for x in X])
    [{'x': 0.4967141530112327},
     {'x': -0.13826430117118466},
     {'x': 0.6476885381006925},
     {'x': 1.5230298564080254},
     {'x': -0.23415337472333597},
     {'x': 0, 'y': 10}]
    '''

    def __init__(self, on, strategy, constant_value=None, aggregate=None):

        self.on = on

        self.allowed_strategies = [
            'mean',
            'max',
            'min',
            'constant',
        ]

        if strategy not in self.allowed_strategies:
            raise ValueError(
                f'Can only use these strategies: {self.allowed_strategies}\
                    got strategy = {strategy}'
            )

        self.functions_dictionnary = {
            'mean': creme.stats.Mean(),
            'max': creme.stats.Max(),
            'min': creme.stats.Min(),
            'constant': Constant(constant_value=constant_value),
        }

        self.impute_function = self.functions_dictionnary[strategy]

    def _impute(self, x):
        x_imputed = {
            i: xi
            for i, xi in x.items()
        }
        x_imputed[self.on] = self.impute_function.get()
        return x_imputed

    def fit_one(self, x):
        if self.on in x:
            self.impute_function.update(x[self.on])
        return self

    def transform_one(self, x):
        if self.on not in x:
            x = self._impute(x)
        return x
