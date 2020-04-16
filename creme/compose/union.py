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


class TransformerUnion(base.Transformer):
    """Packs multiple transformers into a single one.

    Calling `transform_one` will concatenate each transformer's output.

    Parameters:
        transformers: A list of transformers to pack together.

    Attributes:
        transformers (dict)

    Example:

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
        >>> agg = creme.compose.TransformerUnion(mean)
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

    def __init__(self, *transformers):
        self.transformers = {}
        for transformer in transformers:
            self += transformer

    def __getitem__(self, key):
        """Just for convenience."""
        return self.transformers[key]

    def __len__(self):
        """Just for convenience."""
        return len(self.transformers)

    @property
    def is_supervised(self):
        return any(t.is_supervised for t in self.transformers.values())

    def __str__(self):
        return ' + '.join(map(str, self.transformers.values()))

    def __repr__(self):
        return (
            'TransformerUnion (\n\t' +
            '\t'.join(',\n'.join(map(repr, self.transformers.values())).splitlines(True)) +
            '\n)'
        ).expandtabs(2)

    def _get_params(self):
        return dict(self.items())

    def _set_params(self, new_params=None):
        if new_params is None:
            new_params = {}
        return TransformerUnion(*[
            (name, new_params[name])
            if isinstance(new_params.get(name), base.Estimator) else
            (name, step._set_params(new_params.get(name, {})))
            for name, step in self.transformers.items()
        ])

    def add_step(self, transformer):
        """Adds a transformer while taking care of the input type."""

        name = None
        if isinstance(transformer, tuple):
            name, transformer = transformer

        # If the step is a function then wrap it in a FuncTransformer
        if isinstance(transformer, (types.FunctionType, types.LambdaType)):
            transformer = func.FuncTransformer(transformer)

        def infer_name(transformer):
            if isinstance(transformer, func.FuncTransformer):
                return infer_name(transformer.func)
            elif isinstance(transformer, (types.FunctionType, types.LambdaType)):
                return transformer.__name__
            elif hasattr(transformer, '__class__'):
                return transformer.__class__.__name__
            return str(transformer)

        # Infer a name if none is given
        if name is None:
            name = infer_name(transformer)

        if name in self.transformers:
            counter = 1
            while f'{name}{counter}' in self.transformers:
                counter += 1
            name = f'{name}{counter}'

        # Store the transformer
        self.transformers[name] = transformer

        return self

    def __add__(self, other):
        return self.add_step(other)

    def fit_one(self, x, y=None):
        for transformer in self.transformers.values():
            transformer.fit_one(x, y)
        return self

    def transform_one(self, x):
        """Passes the data through each transformer and packs the results together."""
        return dict(collections.ChainMap(*(
            transformer.transform_one(x)
            for transformer in self.transformers.values()
        )))

    def draw(self):

        if not GRAPHVIZ_INSTALLED:
            raise ImportError('graphviz is not installed')

        g = graphviz.Digraph(engine='fdp')

        for part in self.transformers.values():
            if hasattr(part, 'draw'):
                g.subgraph(part.draw())
            else:
                g.node(str(part))
        return g
