import collections
import copy

from . import imputer
from .. import stats


class CategoryImputer(imputer.Imputer):
    """ Imputer allow to replace missing values with descriptive statistics.

    Args:
        on (str): Name of the field to impute.
        by (str): Name of the field to impute with aggregatation.
        how (str): Method to fill missing values.
        k (int): Number of modalities in the target variable.

    >>> import creme

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

    >>> imputer_top_k = creme.impute.CategoryImputer(on='x', exact=True)
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

    >>> imputer_by_town = creme.impute.CategoryImputer(
    ...    on='weather',
    ...    by='town',
    ...    k=10,
    ... )

    >>> X = [
    ...     {'town': 'New York', 'weather': 'sunny'},
    ...     {'town': 'New York', 'weather': 'sunny'},
    ...     {'town': 'New York', 'weather': 'rainy'},
    ...     {'town': 'Montreal', 'weather': 'rainy'},
    ...     {'town': 'Montreal', 'weather': 'humidity'},
    ...     {'town': 'Montreal', 'weather': 'rainy'},
    ...     {'town': 'Pekin', 'weather': 'sunny'},
    ...     {'town': 'New York'},
    ...     {'town': 'Montreal'},
    ...     {'town': 'Pekin'},
    ... ]

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

    def __init__(self, on, by=None, k=25, exact=False):

        super().__init__()
        self.on = on
        self.by = by

        self.imputers = collections.defaultdict(
            lambda: copy.deepcopy(stats.Mode(k=k, exact=exact))
        )
