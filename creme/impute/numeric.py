import collections
import copy

from .. import stats

from . import _constant
from . import imputer


__all__ = ['NumericalImputer']


class NumericImputer(imputer.Imputer):
    """ Imputer allow to replace missing values with descriptive statistics.

    Args:
        on (str): Name of the field to impute.
        by (str): Name of the field to impute with aggregatation.
        how (str): Method to fill missing values.
        constant (float): Constant to replace missing values, when using strategy='constant'.

    Attributes:
        imputer (creme.stats): Object to impute missing values.

    >>> import creme

    >>> X = [
    ...     {'x': 1.0},
    ...     {'x': 2.0},
    ...     {'x': 3.0},
    ...     {}
    ... ]

    >>> imputer_mean = creme.impute.NumericImputer(on='x', how='mean')
    >>> for x in X:
    ...     print(imputer_mean.fit_one(x))
    {'x': 1.0}
    {'x': 2.0}
    {'x': 3.0}
    {'x': 2.0}

    >>> imputer_min = creme.impute.NumericImputer(on='x', how='min')
    >>> for x in X:
    ...     print(imputer_min.fit_one(x))
    {'x': 1.0}
    {'x': 2.0}
    {'x': 3.0}
    {'x': 1.0}

    >>> imputer_max = creme.impute.NumericImputer(on='x', how='max')
    >>> for x in X:
    ...     print(imputer_max.fit_one(x))
    {'x': 1.0}
    {'x': 2.0}
    {'x': 3.0}
    {'x': 3.0}

    >>> imputer_constant = creme.impute.NumericImputer(
    ...     on='x',
    ...     how='constant',
    ...     constant_value=0.0,
    ... )

    >>> for x in X:
    ...     print(imputer_constant.fit_one(x))
    {'x': 1.0}
    {'x': 2.0}
    {'x': 3.0}
    {'x': 0.0}

    >>> imputer_by_shop = creme.impute.NumericImputer(on='customer', by='shop', how='mean')

    >>> X = [
    ...     {'customer': -5.0, 'shop': 'ikea'},
    ...     {'customer': 5.0, 'shop': 'ikea'},
    ...     {'customer': 2.0, 'shop': 'cora'},
    ...     {'customer': 5.0, 'shop': 'cora'},
    ...     {'customer': 5.0, 'shop': 'cora'},
    ...     {'customer': 0.0, 'shop': 'wallmart'},
    ...     {'customer': 10.0, 'shop': 'wallmart'},
    ...     {'shop': 'cora'},
    ...     {'shop': 'ikea'},
    ...     {'shop': 'wallmart'},
    ... ]

    >>> for x in X:
    ...     print(imputer_by_shop.fit_one(x))
    {'customer': -5.0, 'shop': 'ikea'}
    {'customer': 5.0, 'shop': 'ikea'}
    {'customer': 2.0, 'shop': 'cora'}
    {'customer': 5.0, 'shop': 'cora'}
    {'customer': 5.0, 'shop': 'cora'}
    {'customer': 0.0, 'shop': 'wallmart'}
    {'customer': 10.0, 'shop': 'wallmart'}
    {'shop': 'cora', 'customer': 4.0}
    {'shop': 'ikea', 'customer': 0.0}
    {'shop': 'wallmart', 'customer': 5.0}

    """

    def __init__(self, on, by=None, how='mean', constant_value=None):

        super().__init__()
        self.on = on
        self.by = by

        allowed_imputers = {
            'mean': stats.Mean(),
            'max': stats.Max(),
            'min': stats.Min(),
            'constant': _constant.Constant(constant_value=constant_value),
        }

        if how not in allowed_imputers:
            raise ValueError(f'Can only use one of: {allowed_imputers.keys()}')

        self.imputers = collections.defaultdict(
            lambda: copy.deepcopy(allowed_imputers[how]))
