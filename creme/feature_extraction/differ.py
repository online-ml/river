from .. import base


def always_true(x):
    return True


class Differ(base.Transformer):
    """Calculates value differences between different observations.

    Parameters:
        on (str): The name of the feature for which to compute differences. The type of the values
            must support the ``-`` operator, for example an `int` or a `datetime.timedelta`.
        when (callable): A function which indicates when an event occurs. This can used to compute
            the difference since the last time an event occured. If ``True`` then the differences
            will always be computed with regards to the last observation.
        by (str): Can be used to compute value differences for different groups. If ``None`` then
            the differences will be computed globally.
        when_name (str): The name of the of the feature that is returned is automatically built
            from the input parameters. If ``when`` is a ``lambda`` function then no name can be
            inferred. You can thus use this parameter to indicate the name of the event.

    Attributes:
        last_moments (dict): Indicates the last value for each value in the ``by`` field.
        feature_name (str): The name of the resulting feature.

    Example:

        ::

            >>> import datetime as dt

            >>> data = [
            ...     {'weather': 'sunny', 'moment': 1, 'country': 'Sweden'},
            ...     {'weather': 'rainy', 'moment': 1, 'country': 'Rwanda'},
            ...     {'weather': 'rainy', 'moment': 2, 'country': 'Rwanda'},
            ...     {'weather': 'rainy', 'moment': 2, 'country': 'Sweden'},
            ...     {'weather': 'sunny', 'moment': 3, 'country': 'Rwanda'},
            ...     {'weather': 'rainy', 'moment': 4, 'country': 'Rwanda'},
            ...     {'weather': 'rainy', 'moment': 3, 'country': 'Sweden'},
            ...     {'weather': 'sunny', 'moment': 4, 'country': 'Sweden'}
            ... ]

            >>> def is_sunny(x):
            ...     return x['weather'] == 'sunny'

            >>> differ = Differ(
            ...     on='moment',
            ...     by='country',
            ...     when=is_sunny
            ... )

            >>> for x in data:
            ...     differ = differ.fit_one(x)
            ...     print(differ.transform_one(x))
            {'moment_diff_since_is_sunny_by_country': 0}
            {}
            {}
            {'moment_diff_since_is_sunny_by_country': 1}
            {'moment_diff_since_is_sunny_by_country': 0}
            {'moment_diff_since_is_sunny_by_country': 1}
            {'moment_diff_since_is_sunny_by_country': 2}
            {'moment_diff_since_is_sunny_by_country': 0}

    """

    def __init__(self, on, when=True, by=None, when_name=None):
        self.on = on
        self.by = by
        self.when = always_true if when is True else when
        self.when_name = when.__name__ if when_name is None and when is not True else when_name
        self.last_moments = {}
        self.feature_name = f'{self.on}_diff'
        if when is not True:
            self.feature_name += f'_since_{self.when_name}'
        if by is not None:
            self.feature_name += f'_by_{self.by}'

    def fit_one(self, x, y=None):

        if self.when(x):
            self.last_moments[x[self.by]] = x[self.on]

        return self

    def transform_one(self, x):
        if self.when(x):
            return {self.feature_name: x[self.on] - x[self.on]}
        if x[self.by] in self.last_moments:
            return {self.feature_name: x[self.on] - self.last_moments[x[self.by]]}
        return {}
