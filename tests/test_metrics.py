import collections
import functools

from creme import metrics
from sklearn import metrics as sk_metrics


def metric_test(metric, sk_metric, y_true, y_pred):
    for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)):

        if isinstance(y_p, list):
            y_p = {i: p for i, p in enumerate(y_p)}
        metric.update(y_t, y_p)

        if i >= 1:
            assert metric.get() == sk_metric(y_true[:i + 1], y_pred[:i + 1])


def test_log_loss():
    metric_test(
        metric=metrics.LogLoss(),
        sk_metric=sk_metrics.log_loss,
        y_true=[True, False, False, True],
        y_pred=[0.9, 0.1, 0.2, 0.65]
    )


def test_cross_entropy():
    metric_test(
        metric=metrics.CrossEntropy(),
        sk_metric=functools.partial(sk_metrics.log_loss, labels=[0, 1, 2]),
        y_true=[0, 1, 2, 2],
        y_pred=[
            [0.29450637, 0.34216758, 0.36332605],
            [0.21290077, 0.32728332, 0.45981591],
            [0.42860913, 0.33380113, 0.23758974],
            [0.44941979, 0.32962558, 0.22095463]
        ]
    )


def rolling_metric_test(metric, sk_metric, y_true, y_pred):

    def tail(iterable, n):
        return collections.deque(iterable, maxlen=n)

    n = metric.window_size

    for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)):

        if isinstance(y_p, list):
            y_p = {i: p for i, p in enumerate(y_p)}
        metric.update(y_t, y_p)

        if i >= 1:
            assert metric.get() == sk_metric(tail(y_true[:i + 1], n), tail(y_pred[:i + 1], n))


def test_rolling_precision():
    rolling_metric_test(
        metric=metrics.RollingPrecision(3),
        sk_metric=sk_metrics.precision_score,
        y_true=[True, False, True, True, True],
        y_pred=[True, True, False, True, True]
    )


def test_rolling_recall():
    rolling_metric_test(
        metric=metrics.RollingRecall(3),
        sk_metric=sk_metrics.recall_score,
        y_true=[True, False, True, True, True],
        y_pred=[True, True, False, True, True]
    )


def test_rolling_f1():
    rolling_metric_test(
        metric=metrics.RollingF1Score(3),
        sk_metric=sk_metrics.f1_score,
        y_true=[True, False, True, True, True],
        y_pred=[True, True, False, True, True]
    )
