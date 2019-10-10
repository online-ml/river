import math

from . import rmse


__all__ = ['RMSLE']


class RMSLE(rmse.RMSE):
    """Root mean squared logarithmic error.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [3, -0.5, 2, 7]
            >>> y_pred = [2.5, 0.0, 2, 8]

            >>> metric = metrics.RMSLE()
            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     print(metric.update(y_t, y_p).get())
            0.133531...
            0.499141...
            0.407546...
            0.357825...

            >>> metric
            RMSLE: 0.357826

    """

    def update(self, y_true, y_pred, sample_weight=1.):
        return super().update(math.log(y_true + 1), math.log(y_pred + 1), sample_weight)
