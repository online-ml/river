import collections
import types

try:
    import graphviz
    GRAPHVIZ_INSTALLED = True
except ImportError:
    GRAPHVIZ_INSTALLED = False

from .. import base

from . import func


__all__ = ['TransformerUnion']


class TransformerUnion(collections.UserDict, base.Transformer):
    """Packs multiple transformers into a single one.

    Calling ``transform_one`` will concatenate each transformer's output using a
    `collections.ChainMap`.

    Parameters:
        transformers (list): transformers to pack together.

    Example:

        ::

            >>> from pprint import pprint
            >>> import creme.compose
            >>> import creme.feature_extraction
            >>> import creme.stats

            >>> X = [
            ...     {'place': 'Taco Bell', 'revenue': 42},
            ...     {'place': 'Burger King', 'revenue': 16},
            ...     {'place': 'Burger King', 'revenue': 24},
            ...     {'place': 'Taco Bell', 'revenue': 58},
            ...     {'place': 'Burger King', 'revenue': 20},
            ...     {'place': 'Taco Bell', 'revenue': 50}
            ... ]

            >>> mean = creme.feature_extraction.Agg(
            ...     on='revenue',
            ...     by='place',
            ...     how=creme.stats.Mean()
            ... )
            >>> count = creme.feature_extraction.Agg(
            ...     on='revenue',
            ...     by='place',
            ...     how=creme.stats.Count()
            ... )
            >>> agg = creme.compose.TransformerUnion([mean])
            >>> agg += count

            >>> for x in X:
            ...     pprint(agg.fit_one(x).transform_one(x))
            {'revenue_count_by_place': 1, 'revenue_mean_by_place': 42.0}
            {'revenue_count_by_place': 1, 'revenue_mean_by_place': 16.0}
            {'revenue_count_by_place': 2, 'revenue_mean_by_place': 20.0}
            {'revenue_count_by_place': 2, 'revenue_mean_by_place': 50.0}
            {'revenue_count_by_place': 3, 'revenue_mean_by_place': 20.0}
            {'revenue_count_by_place': 3, 'revenue_mean_by_place': 50.0}

            >>> pprint(agg.transform_one({'place': 'Taco Bell'}))
            {'revenue_count_by_place': 3, 'revenue_mean_by_place': 50.0}

    """

    def __init__(self, transformers=None):
        super().__init__()
        if transformers is not None:
            for transformer in transformers:
                self += transformer

    @property
    def is_supervised(self):
        return any(transformer.is_supervised for transformer in self.values())

    def __str__(self):
        """Returns a human friendly representation of the pipeline."""
        return f' + '.join(map(str, self.keys()))

    def __repr__(self):
        return (
            'TransformerUnion (\n    ' +
            '    '.join(',\n'.join(map(repr, self.values())).splitlines(True)) +
            '\n)'
        )

    def add_step(self, other):
        """Adds a transformer while taking care of the input type."""

        # Infer a name if none is given
        if not isinstance(other, (list, tuple)):
            other = (str(other), other)
        name, transformer = other

        # If a function is given then wrap it in a FuncTransformer
        if isinstance(transformer, types.FunctionType):
            name = transformer.__name__
            transformer = func.FuncTransformer(transformer)

        # Prefer clarity to magic
        if name in self:
            raise KeyError(f'{name} already exists')

        # Store the transformer
        self[name] = transformer

        return self

    def __add__(self, other):
        return self.add_step(other)

    def fit_one(self, x, y=None):
        for transformer in self.values():
            transformer.fit_one(x, y)
        return self

    def transform_one(self, x):
        """Passes the data through each transformer and packs the results together."""
        return dict(collections.ChainMap(*(
            transformer.transform_one(x)
            for transformer in self.values()
        )))

    def draw(self):

        if not GRAPHVIZ_INSTALLED:
            raise ImportError('graphviz is not installed')

        g = graphviz.Digraph(engine='fdp')

        for part in self.values():
            if hasattr(part, 'draw'):
                g.subgraph(part.draw())
            else:
                g.node(str(part))
        return g
