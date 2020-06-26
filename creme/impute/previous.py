from creme import base


__all__ = ['PreviousImputer']


class PreviousImputer(base.Transformer):
    """Imputes missing values by using the most recent value.

    Example:

        >>> from creme import impute

        >>> imputer = impute.PreviousImputer()

        >>> imputer = imputer.fit_one({'x': 1, 'y': 2})
        >>> imputer.transform_one({'y': None})
        {'y': 2}

        >>> imputer.transform_one({'x': None})
        {'x': 1}

    """

    def __init__(self):
        self._latest = {}

    def fit_one(self, x):

        for i, v in x.items():
            if v is not None:
                self._latest[i] = v

        return self

    def transform_one(self, x):

        for i, v in x.items():
            if v is None:
                x[i] = self._latest.get(i)

        return x
