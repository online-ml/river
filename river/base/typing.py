from __future__ import annotations

import typing

FeatureName = typing.Hashable
RegTarget = float
ClfTarget = typing.Union[bool, str, int]  # noqa: UP007
Target = typing.Union[ClfTarget, RegTarget]  # noqa: UP007
Dataset = typing.Iterable[typing.Tuple[dict[FeatureName, typing.Any], typing.Any]]  # noqa: UP006
Stream = typing.Iterator[typing.Tuple[dict[FeatureName, typing.Any], typing.Any]]  # noqa: UP006
