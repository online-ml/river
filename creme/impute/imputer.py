import collections
import copy

from . import _constant
from .. import stats

__all__ = ['Imputer']


class Imputer:
    """ Imputer allow to replace missing values with descriptive statistics or any value.

    Parameters:
        on (str): Name of the field to impute.
        by (str): Name of the field to impute with aggregatation.
        stat (stats.Univariate or any value): Univariate statistic used to fill missing
        values, for example: :any:`stats.Mean`. If stat is not instance of stats.Univariate,
        missing values will be replaced by stat value.

    Example:
        >>> import creme

        >>> X = [
        ...     {'x': 1.0},
        ...     {'x': 2.0},
        ...     {'x': 3.0},
        ...     {}
        ... ]

        >>> imputer_constant_value = creme.impute.Imputer(
        ...     on='x',
        ...     stat=5.0
        ... )

        >>> for x in X:
        ...     print(imputer_constant_value.fit_one(x))
        {'x': 1.0}
        {'x': 2.0}
        {'x': 3.0}
        {'x': 5.0}

        >>> imputer_mean = creme.impute.Imputer(
        ...     on='x',
        ...     stat=creme.stats.Mean()
        ... )

        >>> for x in X:
        ...     print(imputer_mean.fit_one(x))
        {'x': 1.0}
        {'x': 2.0}
        {'x': 3.0}
        {'x': 2.0}

        >>> X = [
        ...     {'x': 'sunny'},
        ...     {'x': 'rainy'},
        ...     {'x': 'humidity'},
        ...     {'x': 'sunny'},
        ...     {'x': 'rainy'},
        ...     {'x': 'rainy'},
        ...     {},
        ...     {},
        ...     {},
        ... ]

        >>> imputer_top_k = creme.impute.Imputer(
        ...     on='x',
        ...     stat=creme.stats.Mode(k=25),
        ... )

        >>> for x in X:
        ...     print(imputer_top_k.fit_one(x))
        {'x': 'sunny'}
        {'x': 'rainy'}
        {'x': 'humidity'}
        {'x': 'sunny'}
        {'x': 'rainy'}
        {'x': 'rainy'}
        {'x': 'rainy'}
        {'x': 'rainy'}
        {'x': 'rainy'}

        >>> X = [
        ...   {'town': 'New York', 'weather': 'sunny'},
        ...   {'town': 'New York', 'weather': 'sunny'},
        ...   {'town': 'New York', 'weather': 'rainy'},
        ...   {'town': 'Montreal', 'weather': 'rainy'},
        ...   {'town': 'Montreal', 'weather': 'humidity'},
        ...   {'town': 'Montreal', 'weather': 'rainy'},
        ...   {'town': 'Pekin', 'weather': 'sunny'},
        ...   {'town': 'New York'},
        ...   {'town': 'Montreal'},
        ...   {'town': 'Pekin'},
        ... ]

        >>> imputer_by_town = creme.impute.Imputer(
        ...     on='weather',
        ...     by='town',
        ...     stat=creme.stats.Mode(exact=True)
        ... )

        >>> for x in X:
        ...     print(imputer_by_town.fit_one(x))
        {'town': 'New York', 'weather': 'sunny'}
        {'town': 'New York', 'weather': 'sunny'}
        {'town': 'New York', 'weather': 'rainy'}
        {'town': 'Montreal', 'weather': 'rainy'}
        {'town': 'Montreal', 'weather': 'humidity'}
        {'town': 'Montreal', 'weather': 'rainy'}
        {'town': 'Pekin', 'weather': 'sunny'}
        {'town': 'New York', 'weather': 'sunny'}
        {'town': 'Montreal', 'weather': 'rainy'}
        {'town': 'Pekin', 'weather': 'sunny'}

    """

    def __init__(self, on, stat, by=None):
        self.on = on
        self.by = by

        self.stat = stat if isinstance(
            stat, stats.Univariate) else _constant.Constant(constant_value=stat)

        self.imputers = collections.defaultdict(
            lambda: copy.deepcopy(self.stat))

    def fit_one(self, x):
        if self.on in x:
            key = x[self.by] if self.by else None
            self.imputers[key].update(x[self.on])
            return x
        return self.transform_one(x)

    def transform_one(self, x):
        if self.on not in x:
            key = x[self.by] if self.by else None
            return {
                **x,
                self.on: self.imputers[key].get()
            }
        return x
