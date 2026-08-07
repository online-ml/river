from __future__ import annotations

from river import time_series

__all__ = ["Theta"]


class Theta(time_series.base.Forecaster):
    """Theta forecasting method, unseasonal version.

    The classic Theta model (Assimakopoulos & Nikolopoulos, 2000) was shown
    by Hyndman and Billah (2003) to be equivalent to Simple Exponential
    Smoothing (SES) with a linear drift, where the drift is half the slope
    of an OLS linear trend fitted to the original data.

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
    [3.727, 4.227, 4.727]

    References
    ----------

    [^1]: [Assimakopoulos, V., & Nikolopoulos, K. (2000). The theta model: a
    decomposition approach to forecasting](https://doi.org/10.1016/S0169-2070(00)00052-2).
    International Journal of Forecasting, 16(4), 521-530.

    [^2]: [Hyndman, R. J., & Billah, B. (2003). Unmasking the Theta
    method](https://robjhyndman.com/papers/Theta.pdf). International Journal of
    Forecasting, 19(2), 287-290.
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
        if self.n < 2:
            return [level] * horizon
        denom = self.n * self.sum_tt - self.sum_t**2
        if denom == 0:
            return [level] * horizon
        slope = (self.n * self.sum_ty - self.sum_t * self.sum_y) / denom
        return [level + h * slope / 2.0 for h in range(1, horizon + 1)]