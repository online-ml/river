from __future__ import annotations

from river import time_series

__all__ = ["Theta"]


class Theta(time_series.base.Forecaster):
    """Theta forecasting method, unseasonal version.

    The forecast averages two online components:

    - A Simple Exponential Smoothing level (the flat part).
    - An OLS linear trend on (t, y), kept with running sums and extrapolated
      at half weight (the drift). Averaging the regression line with the SES
      level yields half the regression slope per step, which is the drift the
      original Theta model ends up with once its theta-lines are combined.

    Parameters
    ----------
    alpha
        Smoothing parameter for the SES level.

    Examples
    --------
    >>> from river import time_series
    >>> model = time_series.Theta(alpha=0.3)
    >>> for y in [1, 2, 3, 4, 5]:
    ...     model.learn_one(y)
    >>> [round(f, 3) for f in model.forecast(horizon=3)]
    [4.613, 5.113, 5.613]

    References
    ----------
    Assimakopoulos, V., & Nikolopoulos, K. (2000). The theta model: a
    decomposition approach to forecasting. International Journal of
    Forecasting, 16(4), 521-530.
    
    """

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self.level: float | None = None
        self.n = 0
        self.sum_t = 0.0
        self.sum_y = 0.0
        self.sum_ty = 0.0
        self.sum_tt = 0.0

    @classmethod
    def _unit_test_params(cls):
        yield {"alpha": 0.2}

    def _trend(self, t: int) -> float:
        """OLS fit on (t, y) seen so far, evaluated at t."""
        level = self.level if self.level is not None else 0.0
        if self.n < 2:
            return level
        denom = self.n * self.sum_tt - self.sum_t**2
        if denom == 0:
            return level
        slope = (self.n * self.sum_ty - self.sum_t * self.sum_y) / denom
        intercept = (self.sum_y - slope * self.sum_t) / self.n
        return intercept + slope * t

    def learn_one(self, y: float, x: dict | None = None) -> None:
        self.n += 1
        t = self.n
        self.sum_t += t
        self.sum_y += y
        self.sum_ty += t * y
        self.sum_tt += t * t
        self.level = y if self.level is None else self.alpha * y + (1 - self.alpha) * self.level

    def forecast(self, horizon: int, xs: list[dict] | None = None) -> list:
        level = self.level if self.level is not None else 0.0
        return [0.5 * level + 0.5 * self._trend(self.n + h) for h in range(1, horizon + 1)]
