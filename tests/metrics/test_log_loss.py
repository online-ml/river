from __future__ import annotations

import math

from sklearn import metrics as sk_metrics

from river import metrics


def test_log_loss():
    metric = metrics.LogLoss()

    y_true = [True, False, False, True]
    y_pred = [0.9, 0.1, 0.2, 0.65]

    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        metric.update(yt, yp)

        if i >= 1:
            assert math.isclose(metric.get(), sk_metrics.log_loss(y_true[: i + 1], y_pred[: i + 1]))

    metric.revert(y_true[-1], y_pred[-1])
    assert math.isclose(metric.get(), sk_metrics.log_loss(y_true[:-1], y_pred[:-1]))
