from __future__ import annotations

import typing

FeatureName = typing.Hashable
RegTarget = float
ClfTarget = typing.Union[bool, str, int]  # noqa: UP007
Target = typing.Union[ClfTarget, RegTarget]  # noqa: UP007
Dataset = typing.Iterable[typing.Tuple[dict[FeatureName, typing.Any], typing.Any]]  # noqa: UP006
Stream = typing.Iterator[typing.Tuple[dict[FeatureName, typing.Any], typing.Any]]  # noqa: UP006


# These classes aim to provide the first blocks towards using protocols.
# They should be modified if needed.
class Learner(typing.Protocol):
    def learn_one(self, x: dict[FeatureName, typing.Any], y: Target) -> None: ...


class Predictor(Learner, typing.Protocol):
    def predict_one(self, x: dict[FeatureName, typing.Any]) -> Target: ...
