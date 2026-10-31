from __future__ import annotations

import types
import typing
from collections.abc import Callable, Mapping
from typing import Any

import narwhals.stable.v2 as nw

from river import base, utils

if typing.TYPE_CHECKING:
    from narwhals.stable.v2.typing import IntoDataFrame, IntoSeries


from . import func
from ._routing import method_params, route

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

    >>> import pandas as pd
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
    ...     agg.learn_one(x)
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

    Mini-batch example:

    >>> X = pd.DataFrame([
    ...     {"place": 2, "revenue": 42},
    ...     {"place": 3, "revenue": 16},
    ...     {"place": 3, "revenue": 24},
    ...     {"place": 2, "revenue": 58},
    ...     {"place": 3, "revenue": 20},
    ...     {"place": 2, "revenue": 50},
    ... ])

    Since we need a transformer with mini-batch support to demonstrate, we shall use
    a `StandardScaler`.

    >>> from river import compose
    >>> from river import preprocessing

    >>> agg = (
    ...     compose.Select("place") +
    ...     (compose.Select("revenue") | preprocessing.StandardScaler())
    ... )

    >>> agg.learn_many(X)
    >>> agg.transform_many(X)
       place   revenue
    0      2  0.441250
    1      3 -1.197680
    2      3 -0.693394
    3      2  1.449823
    4      3 -0.945537
    5      2  0.945537

    """

    def __init__(self, *transformers: object) -> None:
        self._transformers: dict[str, base.Estimator] = {}
        for transformer in transformers:
            if type(transformer) is type(self):
                nested = typing.cast(TransformerUnion, transformer)
                for item in nested.transformers.items():
                    self._add_step(item)
            else:
                self._add_step(transformer)
        self._validate()
        self._build_plan()

    @property
    def transformers(self) -> Mapping[str, base.Estimator]:
        return types.MappingProxyType(self._transformers)

    def __getitem__(self, key: str | int) -> base.Estimator:
        """Just for convenience."""
        if isinstance(key, int):
            return self._transformers[list(self._transformers.keys())[key]]
        return self._transformers[key]

    def __len__(self) -> int:
        """Just for convenience."""
        return len(self._transformers)

    def __str__(self) -> str:
        return " + ".join(map(str, self._transformers.values()))

    def __repr__(self) -> str:
        return (
            "TransformerUnion (\n\t"
            + "\t".join(",\n".join(map(repr, self._transformers.values())).splitlines(True))
            + "\n)"
        ).expandtabs(2)

    def _get_params(self) -> dict[str, Any]:
        return {name: transformer._get_params() for name, transformer in self._transformers.items()}

    def clone(
        self, new_params: dict[str, Any] | None = None, include_attributes: bool = False
    ) -> TransformerUnion:
        if new_params is None:
            new_params = {}

        return self.__class__(
            *[
                (name, new_params[name])
                if isinstance(new_params.get(name), base.Estimator)
                else (
                    name,
                    step.clone(new_params.get(name, {}), include_attributes=include_attributes),
                )
                for name, step in self._transformers.items()
            ]
        )

    def mutate(self, new_params: dict[str, Any]) -> None:
        for step_name, step_params in new_params.items():
            self[step_name].mutate(step_params)

    @property
    def _supervised(self) -> bool:
        return any(t._supervised for t in self._transformers.values())

    @property
    def _can_mini_batch(self) -> bool:
        return bool(self._transformers) and all(
            t._can_mini_batch for t in self._transformers.values()
        )

    def _add_step(self, transformer: object) -> TransformerUnion:
        """Adds a transformer while taking care of the input type."""

        name = None
        if isinstance(transformer, tuple):
            name, transformer = typing.cast(tuple[str, object], transformer)

        # If the step is a function then wrap it in a FuncTransformer
        if isinstance(transformer, types.FunctionType) or isinstance(transformer, types.LambdaType):
            transformer = func.FuncTransformer(transformer)

        def infer_name(transformer: object) -> str:
            if type(transformer) is func.FuncTransformer:
                return infer_name(transformer.func)
            elif isinstance(transformer, types.FunctionType) or isinstance(
                transformer, types.LambdaType
            ):
                return transformer.__name__
            elif hasattr(transformer, "__class__"):
                return transformer.__class__.__name__
            return str(transformer)

        # Infer a name if none is given
        if name is None:
            name = infer_name(transformer)

        if name in self._transformers:
            counter = 1
            while f"{name}{counter}" in self._transformers:
                counter += 1
            name = f"{name}{counter}"

        if not isinstance(transformer, base.Estimator):
            raise TypeError(f"{transformer!r} is not an estimator")

        self._transformers[name] = transformer.clone(include_attributes=True)

        return self

    def _validate(self) -> None:
        for name, transformer in self._transformers.items():
            if not isinstance(base._unwrap(transformer), base.Transformer):
                raise TypeError(f"Transformer union step {name!r} does not transform observations")

    def __add__(self, other: object) -> TransformerUnion:
        return self.__class__(*self._transformers.items(), other)

    def _build_plan(self) -> None:
        self._one_before: list[tuple[Callable[..., None], frozenset[str] | None]] = []
        self._one_after: list[tuple[Callable[..., None], frozenset[str] | None]] = []
        self._many_before: list[tuple[Callable[..., None], frozenset[str] | None]] = []
        self._many_after: list[tuple[Callable[..., None], frozenset[str] | None]] = []
        self._transform_one_funcs: list[
            Callable[
                [dict[base.typing.FeatureName, Any]],
                dict[base.typing.FeatureName, Any],
            ]
        ] = []
        self._transform_many_funcs: list[Callable[[IntoDataFrame], IntoDataFrame] | None] = []
        for transformer in self._transformers.values():
            self._transform_one_funcs.append(getattr(transformer, "transform_one"))
            self._transform_many_funcs.append(getattr(transformer, "transform_many", None))
            learn_one = getattr(transformer, "learn_one", None)
            if learn_one is not None:
                target = self._one_after if transformer._supervised else self._one_before
                target.append((learn_one, method_params(transformer, "learn_one")))
            learn_many = getattr(transformer, "learn_many", None)
            if learn_many is not None:
                target = self._many_after if transformer._supervised else self._many_before
                target.append((learn_many, method_params(transformer, "learn_many")))

    def _learn_one_before(
        self,
        x: dict[base.typing.FeatureName, Any],
        params: dict[str, object],
    ) -> None:
        if not params:
            for learn, _ in self._one_before:
                learn(x)
            return
        for learn, accepted in self._one_before:
            learn(x, **route(accepted, params))

    def _learn_one_after(
        self,
        x: dict[base.typing.FeatureName, Any],
        y: object,
        params: dict[str, object],
    ) -> None:
        if not params:
            for learn, _ in self._one_after:
                learn(x, y)
            return
        for learn, accepted in self._one_after:
            learn(x, y, **route(accepted, params))

    def _learn_many_before(self, X: IntoDataFrame, params: dict[str, object]) -> None:
        if not params:
            for learn, _ in self._many_before:
                learn(X)
            return
        for learn, accepted in self._many_before:
            learn(X, **route(accepted, params))

    def _learn_many_after(
        self,
        X: IntoDataFrame,
        y: IntoSeries | None,
        params: dict[str, object],
    ) -> None:
        if not params:
            for learn, _ in self._many_after:
                learn(X, y)
            return
        for learn, accepted in self._many_after:
            learn(X, y, **route(accepted, params))

    def learn_one(
        self,
        x: dict[base.typing.FeatureName, Any],
        y: object = None,
    ) -> None:
        """Update each transformer.

        Parameters
        ----------
        x
            Features.
        y
            An optional target, this is expected to be provided if at least one of the transformers
            is supervised.

        """
        self._learn_one_before(x, {})
        self._learn_one_after(x, y, {})

    def transform_one(
        self, x: dict[base.typing.FeatureName, Any]
    ) -> dict[base.typing.FeatureName, Any]:
        """Passes the data through each transformer and packs the results together."""
        # Equivalent to `dict(ChainMap(*outputs))` (first-occurrence wins for duplicate
        # keys) but avoids the ChainMap construction overhead.
        result: dict[base.typing.FeatureName, Any] = {}
        for transform in reversed(self._transform_one_funcs):
            result.update(transform(x))
        return result

    # Mini-batch methods

    def learn_many(self, X: IntoDataFrame, y: IntoSeries | None = None) -> None:
        """Update each transformer.

        Parameters
        ----------
        X
            Features.
        y
            An optional target, this is expected to be provided if at least one of the transformers
            is supervised.

        """
        self._learn_many_before(X, {})
        self._learn_many_after(X, y, {})

    def transform_many(self, X: IntoDataFrame) -> IntoDataFrame:
        """Passes the data through each transformer and packs the results together.

        The per-transformer outputs are concatenated column-wise via narwhals, so the result is
        rebuilt in the caller's own backend (pandas, polars, pyarrow, ...). All transformers see
        the same rows in the same order, so a positional horizontal concat matches the historical
        index-aligned `pandas.concat(axis=1)`.
        """
        frames = []
        for transform in self._transform_many_funcs:
            if transform is None:
                raise AttributeError("A union step does not implement transform_many")
            frames.append(utils.dataframe.into_frame(transform(X)))
        return typing.cast("IntoDataFrame", nw.concat(frames, how="horizontal").to_native())
