from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from narwhals.stable.v2.typing import IntoDataFrame, IntoSeries

FeatureName = typing.Hashable
RegTarget = float
ClfTarget = typing.Union[bool, str, int]  # noqa: UP007
Target = typing.Union[ClfTarget, RegTarget]  # noqa: UP007
Dataset = typing.Iterable[typing.Tuple[dict[FeatureName, typing.Any], typing.Any]]  # noqa: UP006
Stream = typing.Iterator[typing.Tuple[dict[FeatureName, typing.Any], typing.Any]]  # noqa: UP006


@typing.runtime_checkable
class Transformer(typing.Protocol):
    @property
    def _supervised(self) -> bool: ...

    def transform_one(self, x: dict[FeatureName, typing.Any]) -> dict[FeatureName, typing.Any]: ...


@typing.runtime_checkable
class MiniBatchTransformer(typing.Protocol):
    @property
    def _supervised(self) -> bool: ...

    def transform_many(self, X: IntoDataFrame) -> IntoDataFrame: ...


@typing.runtime_checkable
class UnsupervisedLearner(typing.Protocol):
    def learn_one(self, x: dict[FeatureName, typing.Any], **kwargs: object) -> None: ...


@typing.runtime_checkable
class Learner(typing.Protocol):
    def learn_one(self, x: dict[FeatureName, typing.Any], y: Target, **kwargs: object) -> None: ...


@typing.runtime_checkable
class MiniBatchLearner(typing.Protocol):
    def learn_many(
        self, X: IntoDataFrame, y: IntoSeries | None = None, **kwargs: object
    ) -> None: ...


@typing.runtime_checkable
class Predictor(typing.Protocol):
    def predict_one(self, x: dict[FeatureName, typing.Any], **kwargs: object) -> Target | None: ...


@typing.runtime_checkable
class ProbabilisticPredictor(typing.Protocol):
    def predict_proba_one(
        self, x: dict[FeatureName, typing.Any], **kwargs: object
    ) -> dict[ClfTarget, float]: ...


@typing.runtime_checkable
class MiniBatchPredictor(typing.Protocol):
    def predict_many(self, X: IntoDataFrame, **kwargs: object) -> IntoSeries: ...


@typing.runtime_checkable
class MiniBatchProbabilisticPredictor(typing.Protocol):
    def predict_proba_many(self, X: IntoDataFrame, **kwargs: object) -> IntoDataFrame: ...


# The following protocols enable us to request that objects can be ordered.
# They are equivalent to typeshed's SupportsRichComparison.
class SupportsComparisonGreater(typing.Protocol):
    def __gt__(self, other: typing.Any, /) -> bool: ...


class SupportsComparisonLesser(typing.Protocol):
    def __lt__(self, other: typing.Any, /) -> bool: ...


SupportsComparison = SupportsComparisonGreater | SupportsComparisonLesser
