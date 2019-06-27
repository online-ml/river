from . import mse


__all__ = ['RMSE', 'RollingRMSE']


class RMSE(mse.MSE):
    """Root mean squared error.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [3, -0.5, 2, 7]
            >>> y_pred = [2.5, 0.0, 2, 8]

            >>> metric = metrics.RMSE()
            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     print(metric.update(y_t, y_p).get())
            0.5
            0.5
            0.408248...
            0.612372...

            >>> metric
            RMSE: 0.612372

    """

    def get(self):
        return super().get() ** 0.5


class RollingRMSE(mse.RollingMSE):
    """Rolling root mean squared error.

    Parameters:
        window_size (int): Size of the window of recent values to consider.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [3, -0.5, 2, 7]
            >>> y_pred = [2.5, 0.0, 2, 8]

            >>> metric = metrics.RollingRMSE(window_size=2)
            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     print(metric.update(y_t, y_p))
            RollingRMSE: 0.5
            RollingRMSE: 0.5
            RollingRMSE: 0.353553
            RollingRMSE: 0.707107

    """

    def get(self):
        return super().get() ** 0.5
