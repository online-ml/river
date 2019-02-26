from collections import defaultdict

from . import base


class CategoricalCount(base.RunningStatistic):
    """Counts the running number of categorical value.

    Attributes:
        cat_count (dict) : running number of categorical value.

    Example:

    ::
        >>> from creme import stats
        >>> test = ['Rob', 'Abdil', 'Raph', 'Geo', 'Rob', None, 'Max', 'My life is creme']
        >>> cat_count = stats.CategoricalCount()
        >>> for val in test:
        ...     _ = cat_count.update(val)
        ... 
        >>> print(cat_count.get())
        {'Rob': 2, 'Abdil': 1, 'Raph': 1, 'Geo': 1, None: 1, 'Max': 1, 'My life is creme': 1}

    """

    def __init__(self):
        self.cat_count = defaultdict(int)

    @property
    def name(self):
        return 'categorical_count'

    def update(self, x=None):
        self.cat_count[x] += 1
        return self

    def get(self):
        return dict(self.cat_count)
