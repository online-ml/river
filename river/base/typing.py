from __future__ import annotations

import numbers
import typing

FeatureName = typing.Hashable
RegTarget = numbers.Number
ClfTarget = bool | str | int
Target = ClfTarget | RegTarget
Dataset = typing.Iterable[tuple[dict, typing.Any]]
Stream = typing.Iterator[tuple[dict, typing.Any]]
