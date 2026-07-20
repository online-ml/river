from __future__ import annotations

from river import stats


class Count(stats.base.Statistic[int]):
    """A simple counter.

    Attributes
    ----------
    n : int
        The current number of observations.

    """

    def __init__(self) -> None:
        self.n: int = 0

    def update(self) -> None:
        self.n += 1

    def get(self) -> int:
        return self.n
