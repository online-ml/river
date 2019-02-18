import collections
import copy

from . import imputer
from . import _top_k


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

    >>> imputer_top_k = creme.impute.CategoryImputer(on='x', how='top_k', k=6)
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
    ...    how='top_k',
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
    ...     try :
    ...         print(imputer_by_town.fit_one(x))
    ...     except:
    ...         print(x)
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

    def __init__(self, on, by=None, how='top_k', k=None):

        super().__init__()

        self.on = on

        self.by = by

        allowed_imputers = {
            'top_k': _top_k.TopK(k=k),
        }

        if how not in allowed_imputers:
            raise ValueError(f'Can only use one of: {allowed_imputers.keys()}')

        self.imputers = collections.defaultdict(
            lambda: copy.deepcopy(allowed_imputers[how]))
