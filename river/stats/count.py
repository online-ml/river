from __future__ import annotations

from typing import Any

from river import stats


class Count(stats.base.Univariate[Any, int]):
    """A simple counter.

    Attributes
    ----------
    n : int
        The current number of observations.

    """

    def __init__(self) -> None:
        self.n: int = 0

    def update(self, x: Any = None) -> None:
        self.n += 1

    def get(self) -> int:
        return self.n
