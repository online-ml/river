from .. import stats

from . import base


__all__ = ['MSE', 'RollingMSE']


class MSE(stats.Mean, base.RegressionMetric):
    """Mean squared error.

    Example:

        ::

            >>> import math
            >>> from creme import metrics
            >>> from sklearn.metrics import mean_squared_error

            >>> y_true = [3, -0.5, 2, 7]
            >>> y_pred = [2.5, 0.0, 2, 8]

            >>> metric = metrics.MSE()
            >>> for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
            ...     metric = metric.update(y_t, y_p)
            ...     assert math.isclose(
            ...         metric.get(),
            ...         mean_squared_error(y_true[:i+1], y_pred[:i+1])
            ...     )

            >>> metric
            MSE: 0.375

    """

    def update(self, y_true, y_pred):
        return super().update((y_true - y_pred) ** 2)


class RollingMSE(stats.RollingMean, base.RegressionMetric):
    """Rolling mean squared error.

    Parameters:
        window_size (int): Size of the window of recent values to consider.

    Example:

        ::

            >>> import math
            >>> from creme import metrics
            >>> from sklearn.metrics import mean_squared_error

            >>> y_true = [3, -0.5, 2, 7]
            >>> y_pred = [2.5, 0.0, 2, 8]

            >>> metric = metrics.RollingMSE(window_size=2)
            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     print(metric.update(y_t, y_p).get())
            0.25
            0.25
            0.125
            0.5

    """

    def update(self, y_true, y_pred):
        return super().update((y_true - y_pred) ** 2)
