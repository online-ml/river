import collections
import copy
import functools

from .. import stats


__all__ = ['StatImputer']


class Constant(stats.Univariate):
    """Implements the ``stats.Univariate`` interface but always returns the same value.

    Arguments:
        value (any): A value

    """

    def __init__(self, value):
        self.value = value

    def update(self, x):
        return self

    def get(self):
        return self.value

    @property
    def name(self):
        return self.value


class StatImputer:
    """Imputer that allows to replace missing values with a univariate statistic, or a constant.

    Parameters:
        on (str): Name of the field to impute.
        by (str): Name of the field to impute with aggregatation.
        stat (stats.Univariate or any value): Univariate statistic used to fill missing values, for
            `stats.Mean`. If ``stat`` is not an instance of `stats.Univariate`, then the input will
            be treated as a constant and all missing values will be replaced by it.

    Example:

        ::

            >>> from creme import impute
            >>> from creme import stats

            >>> X = [
            ...     {'x': 1.0},
            ...     {'x': 2.0},
            ...     {'x': 3.0},
            ...     {}
            ... ]

            >>> const_imp = impute.StatImputer(
            ...     on='x',
            ...     stat=5.0
            ... )

            >>> for x in X:
            ...     print(const_imp.fit_one(x))
            {'x': 1.0}
            {'x': 2.0}
            {'x': 3.0}
            {'x': 5.0}

            >>> mean_imp = impute.StatImputer(
            ...     on='x',
            ...     stat=stats.Mean()
            ... )

            >>> for x in X:
            ...     print(mean_imp.fit_one(x))
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

            >>> mode_imp = impute.StatImputer(
            ...     on='x',
            ...     stat=stats.Mode(),
            ... )

            >>> for x in X:
            ...     print(mode_imp.fit_one(x))
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

            >>> by_town_imp = impute.StatImputer(
            ...     on='weather',
            ...     by='town',
            ...     stat=stats.Mode()
            ... )

            >>> for x in X:
            ...     print(by_town_imp.fit_one(x))
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
        self.stat = stat if isinstance(stat, stats.Univariate) else Constant(stat)
        self.imputers = collections.defaultdict(functools.partial(copy.deepcopy, self.stat))

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
