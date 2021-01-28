import collections
import types

from river import base

from . import func

__all__ = ["TransformerUnion"]


class TransformerUnion(base.Transformer):
    """Packs multiple transformers into a single one.

    Pipelines allow you to apply steps sequentially. Therefore, the output of a step becomes the
    input of the next one. In many cases, you may want to pass the output of a step to multiple
    steps. This simple transformer allows you to do so. In other words, it enables you to apply
    particular steps to different parts of an input. A typical example is when you want to scale
    numeric features and one-hot encode categorical features.

    This transformer is essentially a list of transformers. Whenever it is updated, it loops
    through each transformer and updates them. Meanwhile, calling `transform_one` collects the
    output of each transformer and merges them into a single dictionary.

    Parameters
    ----------
    transformers
        Ideally, a list of (name, estimator) tuples. A name is automatically inferred if none is
        provided.

    Examples
    --------

    Take the following dataset:

    >>> X = [
    ...     {'place': 'Taco Bell', 'revenue': 42},
    ...     {'place': 'Burger King', 'revenue': 16},
    ...     {'place': 'Burger King', 'revenue': 24},
    ...     {'place': 'Taco Bell', 'revenue': 58},
    ...     {'place': 'Burger King', 'revenue': 20},
    ...     {'place': 'Taco Bell', 'revenue': 50}
    ... ]

    As an example, let's assume we want to compute two aggregates of a dataset. We therefore
    define two `feature_extraction.Agg`s and initialize a `TransformerUnion` with them:

    >>> from river import compose
    >>> from river import feature_extraction
    >>> from river import stats

    >>> mean = feature_extraction.Agg(
    ...     on='revenue', by='place',
    ...     how=stats.Mean()
    ... )
    >>> count = feature_extraction.Agg(
    ...     on='revenue', by='place',
    ...     how=stats.Count()
    ... )
    >>> agg = compose.TransformerUnion(mean, count)

    We can now update each transformer and obtain their output with a single function call:

    >>> from pprint import pprint
    >>> for x in X:
    ...     agg = agg.learn_one(x)
    ...     pprint(agg.transform_one(x))
    {'revenue_count_by_place': 1, 'revenue_mean_by_place': 42.0}
    {'revenue_count_by_place': 1, 'revenue_mean_by_place': 16.0}
    {'revenue_count_by_place': 2, 'revenue_mean_by_place': 20.0}
    {'revenue_count_by_place': 2, 'revenue_mean_by_place': 50.0}
    {'revenue_count_by_place': 3, 'revenue_mean_by_place': 20.0}
    {'revenue_count_by_place': 3, 'revenue_mean_by_place': 50.0}

    Note that you can use the `+` operator as a shorthand notation:

    agg = mean + count

    This allows you to build complex pipelines in a very terse manner. For instance, we can
    create a pipeline that scales each feature and fits a logistic regression as so:

    >>> from river import linear_model as lm
    >>> from river import preprocessing as pp

    >>> model = (
    ...     (mean + count) |
    ...     pp.StandardScaler() |
    ...     lm.LogisticRegression()
    ... )

    Whice is equivalent to the following code:

    >>> model = compose.Pipeline(
    ...     compose.TransformerUnion(mean, count),
    ...     pp.StandardScaler(),
    ...     lm.LogisticRegression()
    ... )

    Note that you access any part of a `TransformerUnion` by name:

    >>> model['TransformerUnion']['Agg']
    Agg (
        on="revenue"
        by=['place']
        how=Mean ()
    )

    >>> model['TransformerUnion']['Agg1']
    Agg (
        on="revenue"
        by=['place']
        how=Count ()
    )

    You can also manually provide a name for each step:

    >>> agg = compose.TransformerUnion(
    ...     ('Mean revenue by place', mean),
    ...     ('# by place', count)
    ... )

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

    def __str__(self):
        return " + ".join(map(str, self.transformers.values()))

    def __repr__(self):
        return (
            "TransformerUnion (\n\t"
            + "\t".join(
                ",\n".join(map(repr, self.transformers.values())).splitlines(True)
            )
            + "\n)"
        ).expandtabs(2)

    def _get_params(self):
        return {
            name: transformer._get_params()
            for name, transformer in self.transformers.items()
        }

    def _set_params(self, new_params: dict = None):

        if new_params is None:
            new_params = {}

        return TransformerUnion(
            *[
                (name, new_params[name])
                if isinstance(new_params.get(name), base.Estimator)
                else (name, step._set_params(new_params.get(name, {})))
                for name, step in self.transformers.items()
            ]
        )

    @property
    def _supervised(self):
        return any(t._supervised for t in self.transformers.values())

    def _add_step(self, transformer):
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
            elif hasattr(transformer, "__class__"):
                return transformer.__class__.__name__
            return str(transformer)

        # Infer a name if none is given
        if name is None:
            name = infer_name(transformer)

        if name in self.transformers:
            counter = 1
            while f"{name}{counter}" in self.transformers:
                counter += 1
            name = f"{name}{counter}"

        # Store the transformer
        self.transformers[name] = transformer

        return self

    def __add__(self, other):
        return self._add_step(other)

    def learn_one(self, x, y=None):
        """Update each transformer.

        Parameters
        ----------
        x
            Features.
        y
            An optional target, this is expected to be provided if at least one of the transformers
            is supervised (i.e. it inherits from `base.SupervisedTransformer`).

        """
        for t in self.transformers.values():
            if isinstance(t, base.SupervisedTransformer):
                t.learn_one(x, y)
            else:
                t.learn_one(x)
        return self

    def transform_one(self, x):
        """Passes the data through each transformer and packs the results together."""
        return dict(
            collections.ChainMap(
                *(t.transform_one(x) for t in self.transformers.values())
            )
        )
