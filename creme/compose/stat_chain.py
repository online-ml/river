
import collections
import types
import typing

from creme import stats

from . import func
from . import union


__all__ = ['StatChain']


class StatChain:
    """Chains a sequence of statistics.

    Sequentially apply a list of statistics.

    Parameters:
        statistics: Ideally a list of (name, statistic) tuples. If a statistic is given without a name,
            then a name is automatically inferred from the statistics.

    Attributes:
        statistics (collections.OrderedDict)

        >>> from pprint import pprint
        >>> import creme.compose
        >>> import creme.feature_extraction
        >>> import creme.stats

        >>> X = [
        ...     1,
        ...     2,
        ...     3,
        ...     4,
        ...     5,
        ...     6,
        ... ]

        >>> statistics = creme.stats.Shift(2) | creme.stats.RollingMean(2)

        >>> statistics
        Shift: 0.
        RollingMean: 0.

        >>> for x in X:
        ...     pprint(statistics.update(x).get())
            0.0
            0.0
            0.5
            1.5
            2.5
            3.5

        >>> print(statistics.name)
        shift_2_rolling_rollingmean_2

    """

    def __init__(self, *statistics):
        self.statistics = collections.OrderedDict()
        for stat in statistics:
            self |= stat

    def __getitem__(self, key):
        """Just for convenience."""
        return self.statistics[key]

    def __len__(self):
        """Just for convenience."""
        return len(self.statistics)

    def __str__(self):
        return ' + '.join(map(str, self.statistics.values()))

    @property
    def name(self):
        return '_'.join(self.statistics.keys())

    def __repr__(self):
        return (''.join('\n'.join(map(repr, self.statistics.values())).splitlines(True))
            ).expandtabs(2)

    def _get_params(self):
        return dict(self.items())

    def _set_params(self, new_params=None):
        if new_params is None:
            new_params = {}
        return StatChain(*[
            (name, new_params[name])
            if isinstance(new_params.get(name), stats.Univariate) else
            (name, step._set_params(new_params.get(name, {})))
            for name, step in self.statistics.items()
        ])


    def _add_step(self, stat):
        """Adds a stat while taking care of the input type."""

        name = None
        if isinstance(stat, tuple):
            name, stat = stat

        # If the step is a function then wrap it in a FuncTransformer
        if isinstance(stat, (types.FunctionType, types.LambdaType)):
            stat = func.FuncStat(stat)

        def infer_name(stat):
            if isinstance(stat, func.FuncStat):
                return infer_name(stat.func)
            elif isinstance(stat, stats.base.Statistic):
                return stat.name
            elif hasattr(stat, '__class__'):
                return stat.__class__.__name__
            return str(stat)

        # Infer a name if none is given
        if name is None:
            name = infer_name(stat)

        if name in self.statistics:
            counter = 1
            while f'{name}{counter}' in self.statistics:
                counter += 1
            name = f'{name}{counter}'

        # Store the stat
        self.statistics[name] = stat

        return self

    def __or__(self, other):
        """Inserts a step at the end of the chain."""
        self._add_step(other)
        return self

    def update(self, x):
        for stat in self.statistics.values():
            x = stat.update(x).get()
        return self

    def get(self):
        """Get the value of the last statistic of the ChainStat"""
        return self.statistics[next(reversed(self.statistics))].get()
