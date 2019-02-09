import creme.stats

from .constant import Constant


class NumericalImputer:
    '''
    Imputer allow to replace missing values with descriptive statistics.

    Args:
        missing_values (string): Shape of the missing values.
        strategy (string): Name of the function to replace missing values.
        constant (float): Constant to replace missing values.

    Attributes:
        missing_values (string): Shape of the missing values.
        allowed_strategies (list): List of allowed strategies to impute missing values.
        functions_dictionnary (dic): Mapping between allowed strategies and creme functions.
        impute_function (creme.stats): Function to impute missing values.

    >>> import pprint
    >>> import creme
    >>> import numpy as np
    >>> from sklearn import preprocessing

    >>> np.random.seed(42)
    >>> X = [{'x': v} for v in np.random.normal(loc=0, scale=1, size=5)]
    >>> X.append({'x': 'NaN'})

    >>> imputer_mean = creme.imputer.NumericalImputer(strategy='mean', missing_values='NaN')
    >>> pprint.pprint([imputer_mean.impute(x) for x in X])
    [{'x': 0.4967141530112327},
     {'x': -0.13826430117118466},
     {'x': 0.6476885381006925},
     {'x': 1.5230298564080254},
     {'x': -0.23415337472333597},
     {'x': 0.45900297432508597}]

    >>> imputer_min = creme.imputer.NumericalImputer(strategy='min', missing_values='NaN')
    >>> pprint.pprint([imputer_min.impute(x) for x in X])
    [{'x': 0.4967141530112327},
     {'x': -0.13826430117118466},
     {'x': 0.6476885381006925},
     {'x': 1.5230298564080254},
     {'x': -0.23415337472333597},
     {'x': -0.23415337472333597}]

    >>> imputer_min = creme.imputer.NumericalImputer(strategy='max', missing_values='NaN')

    >>> pprint.pprint([imputer_min.impute(x) for x in X])
    [{'x': 0.4967141530112327},
     {'x': -0.13826430117118466},
     {'x': 0.6476885381006925},
     {'x': 1.5230298564080254},
     {'x': -0.23415337472333597},
     {'x': 1.5230298564080254}]

    >>> imputer_min = creme.imputer.NumericalImputer(strategy='constant', missing_values='NaN', constant_value=0)

    >>> pprint.pprint([imputer_min.impute(x) for x in X])
    [{'x': 0.4967141530112327},
     {'x': -0.13826430117118466},
     {'x': 0.6476885381006925},
     {'x': 1.5230298564080254},
     {'x': -0.23415337472333597},
     {'x': 0}]
    '''

    def __init__(self, strategy, missing_values="NaN", constant_value=None, aggregate=None):

        self.missing_values = missing_values

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

        self.impute_function = self.functions_dictionnary[
            strategy
        ]

    def impute(self, x):
        for i, xi in x.items():
            if xi == self.missing_values:
                imputed_value = self._get(i)
            else:
                imputed_value = self._update(i=i, xi=xi)
        return imputed_value

    def _update(self, i, xi):
        self.impute_function.update(xi)
        return {
            i: xi
        }

    def _get(self, i):
        return {
            i: self.impute_function.get()
        }
