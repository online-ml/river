from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class Interval:
    """An object to represent a (prediction) interval.

    Users are not expected to use this class as-is. Instead, they should use the `with_interval`
    parameter of the `predict_one` method of any regressor or classifier wrapped with a conformal
    prediction method.

    Parameters
    ----------
    lower
        The lower bound of the interval.
    upper
        The upper bound of the interval.

    """

    lower: float
    upper: float

    @property
    def center(self):
        """The center of the interval."""
        return (self.lower + self.upper) / 2

    @property
    def width(self):
        """The width of the interval."""
        return self.upper - self.lower

    def __contains__(self, x):
        return self.lower <= x <= self.upper
