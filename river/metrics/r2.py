from river import stats

from . import base

__all__ = ["R2"]


class R2(base.RegressionMetric):
    """Coefficient of determination ($R^2$) score

    The coefficient of determination, denoted $R^2$ or $r^2$, is the proportion
    of the variance in the dependent variable that is predictable from the
    independent variable(s). [^1]

    Best possible score is 1.0 and it can be negative (because the model can be
    arbitrarily worse). A constant model that always predicts the expected
    value of $y$, disregarding the input features, would get a $R^2$ score of
    0.0.

    $R^2$ is not defined when less than 2 samples have been observed. This
    implementation returns 0.0 in this case.

    Examples
    --------
    >>> from river import metrics

    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]

    >>> metric = metrics.R2()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp).get())
    0.0
    0.9183
    0.9230
    0.9486

    References
    ----------
    [^1]: [Coefficient of determination (Wikipedia)](https://en.wikipedia.org/wiki/Coefficient_of_determination)

    """

    def __init__(self):
        super().__init__()
        self._y_var = stats.Var()
        self._total_sum_of_squares = 0
        self._residual_sum_of_squares = 0
        self.sample_correction = {}

    @property
    def bigger_is_better(self):
        return True

    def update(self, y_true, y_pred, sample_weight=1.0):
        self._y_var.update(y_true, w=sample_weight)
        squared_error = (y_true - y_pred) * (y_true - y_pred) * sample_weight
        self._residual_sum_of_squares += squared_error

        # To track back
        self.sample_correction = {"squared_error": squared_error}

        return self

    def revert(self, y_true, y_pred, sample_weight, correction=None):
        self._y_var.update(y_true, w=-sample_weight)
        self._residual_sum_of_squares -= correction["squared_error"]

        return self

    def get(self):
        if self._y_var.mean.n > 1:
            try:
                total_sum_of_squares = (self._y_var.mean.n - 1) * self._y_var.get()
                return 1 - (self._residual_sum_of_squares / total_sum_of_squares)
            except ZeroDivisionError:
                return 0.0

        # Not defined for n_samples < 2
        return 0.0
