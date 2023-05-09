from __future__ import annotations

import numbers
import typing

FeatureName = typing.Hashable
RegTarget = numbers.Number
ClfTarget = typing.Union[bool, str, int]  # noqa: UP007
Target = typing.Union[ClfTarget, RegTarget]  # noqa: UP007
Dataset = typing.Iterable[typing.Tuple[dict, typing.Any]]  # noqa: UP006
Stream = typing.Iterator[typing.Tuple[dict, typing.Any]]  # noqa: UP006
