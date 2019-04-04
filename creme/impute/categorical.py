import collections
import copy

from . import imputer
from . import _constant

from .. import stats


class CategoricalImputer(imputer.Imputer):
    """ Imputer allow to replace missing values with descriptive statistics.

    Parameters:
        on (str): Name of the field to impute.
        by (str): Name of the field to impute with aggregatation.
        how (str): Method to fill missing values.
        k (int): Number of modalities in the target variable.
        constant_value (str): Constant to replace missing values, when using strategy='constant'.

    Example:

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

        >>> imputer_top_k = creme.impute.CategoricalImputer(on='x', exact=True)
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

        >>> imputer_top_k = creme.impute.CategoricalImputer(
        ...     on='x',
        ...     how='constant',
        ...     constant_value='unknown',
        ... )

        >>> for x in X:
        ...     print(imputer_top_k.fit_one(x))
        {'x': 'sunny'}
        {'x': 'rainy'}
        {'x': 'humidity'}
        {'x': 'sunny'}
        {'x': 'rainy'}
        {'x': 'rainy'}
        {'x': 'unknown'}
        {'x': 'unknown'}
        {'x': 'unknown'}

        >>> imputer_by_town = creme.impute.CategoricalImputer(
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

    def __init__(self, on, how='mode', by=None, k=25, exact=False, constant_value=None):

        super().__init__()
        self.on = on
        self.by = by

        allowed_imputers = {
            'mode': stats.Mode(),
            'constant': _constant.Constant(constant_value=constant_value),
        }

        if how not in allowed_imputers:
            raise ValueError(
                f'Can only use one of: {allowed_imputers.keys()}')

        self.imputers = collections.defaultdict(
            lambda: copy.deepcopy(allowed_imputers[how]))
