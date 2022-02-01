import math

from sklearn import metrics as sk_metrics

from river import metrics


def test_cross_entropy():

    metric = metrics.CrossEntropy()

    y_true = [0, 1, 2, 2]
    y_pred = [
        [0.29450637, 0.34216758, 0.36332605],
        [0.21290077, 0.32728332, 0.45981591],
        [0.42860913, 0.33380113, 0.23758974],
        [0.44941979, 0.32962558, 0.22095463],
    ]

    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        yp = dict(enumerate(yp))
        metric.update(yt, yp)

        if i >= 1:
            assert math.isclose(
                metric.get(),
                sk_metrics.log_loss(y_true[: i + 1], y_pred[: i + 1], labels=[0, 1, 2]),
            )

    metric.revert(y_true[-1], dict(enumerate(y_pred[-1])))
    assert math.isclose(metric.get(), sk_metrics.log_loss(y_true[:-1], y_pred[:-1]))
