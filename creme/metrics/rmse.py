from . import mse


class RMSE(mse.MSE):
    """Exact root mean squared error.

    Example:

    ::

        >>> from creme import metrics

        >>> y_true = [3, -0.5, 2, 7]
        >>> y_pred = [2.5, 0.0, 2, 8]

        >>> mse = metrics.RMSE()
        >>> for y_t, y_p in zip(y_true, y_pred):
        ...     print(mse.update(y_t, y_p).get())
        0.5
        0.5
        0.408248...
        0.612372...

    """

    def get(self):
        return super().get() ** 0.5
