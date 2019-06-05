import collections
import functools
import math

import pytest

from creme import metrics
from sklearn import metrics as sk_metrics


@pytest.mark.parametrize(
    'metric, sk_metric, y_true, y_pred',
    [
        (
            metrics.Precision(),
            sk_metrics.precision_score,
            [True, False, True, True, True],
            [True, True, False, True, True]
        ),
        (
            metrics.MacroPrecision(),
            functools.partial(sk_metrics.precision_score, average='macro'),
            [0, 1, 2, 2, 2],
            [0, 0, 2, 2, 1]
        ),
        (
            metrics.MicroPrecision(),
            functools.partial(sk_metrics.precision_score, average='micro'),
            [0, 1, 2, 2, 2],
            [0, 0, 2, 2, 1]
        ),

        (
            metrics.Recall(),
            sk_metrics.recall_score,
            [True, False, True, True, True],
            [True, True, False, True, True]
        ),
        (
            metrics.MacroRecall(),
            functools.partial(sk_metrics.recall_score, average='macro'),
            [0, 1, 2, 2, 2],
            [0, 0, 2, 2, 1]
        ),
        (
            metrics.MicroRecall(),
            functools.partial(sk_metrics.recall_score, average='micro'),
            [0, 1, 2, 2, 2],
            [0, 0, 2, 2, 1]
        ),

        (
            metrics.F1Score(),
            sk_metrics.f1_score,
            [True, False, True, True, True],
            [True, True, False, True, True]
        ),
        (
            metrics.MacroF1Score(),
            functools.partial(sk_metrics.f1_score, average='macro'),
            [0, 1, 2, 2, 2],
            [0, 0, 2, 2, 1]
        ),
        (
            metrics.MicroF1Score(),
            functools.partial(sk_metrics.f1_score, average='micro'),
            [0, 1, 2, 2, 2],
            [0, 0, 2, 2, 1]
        ),

        (
            metrics.LogLoss(),
            sk_metrics.log_loss,
            [True, False, False, True],
            [0.9, 0.1, 0.2, 0.65]
        ),
        (
            metrics.CrossEntropy(),
            functools.partial(sk_metrics.log_loss, labels=[0, 1, 2]),
            [0, 1, 2, 2],
            [
                [0.29450637, 0.34216758, 0.36332605],
                [0.21290077, 0.32728332, 0.45981591],
                [0.42860913, 0.33380113, 0.23758974],
                [0.44941979, 0.32962558, 0.22095463]
            ]
        )
    ]
)
def test_metric(metric, sk_metric, y_true, y_pred):

    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):

        if isinstance(yp, list):
            yp = dict(enumerate(yp))

        metric.update(yt, yp)

        if i >= 1:
            assert math.isclose(metric.get(), sk_metric(y_true[:i + 1], y_pred[:i + 1]))


@pytest.mark.parametrize(
    'metric, sk_metric, y_true, y_pred',
    [
        (
            metrics.RollingPrecision(3),
            sk_metrics.precision_score,
            [True, False, True, True, True],
            [True, True, False, True, True]
        ),
        (
            metrics.RollingMacroPrecision(3),
            functools.partial(sk_metrics.precision_score, average='macro'),
            [0, 1, 2, 2, 2],
            [0, 0, 2, 2, 1]
        ),
        (
            metrics.RollingMicroPrecision(3),
            functools.partial(sk_metrics.precision_score, average='micro'),
            [0, 1, 2, 2, 2],
            [0, 0, 2, 2, 1]
        ),
        (
            metrics.RollingRecall(3),
            sk_metrics.recall_score,
            [True, False, True, True, True],
            [True, True, False, True, True]
        ),
        (
            metrics.RollingMacroRecall(3),
            functools.partial(sk_metrics.recall_score, average='macro'),
            [0, 1, 2, 2, 2],
            [0, 0, 2, 2, 1]
        ),
        (
            metrics.RollingMicroRecall(3),
            functools.partial(sk_metrics.recall_score, average='micro'),
            [0, 1, 2, 2, 2],
            [0, 0, 2, 2, 1]
        ),
        (
            metrics.RollingF1Score(3),
            sk_metrics.f1_score,
            [True, False, True, True, True],
            [True, True, False, True, True]
        )
    ]
)
def test_rolling_metric(metric, sk_metric, y_true, y_pred):

    def tail(iterable, n):
        return collections.deque(iterable, maxlen=n)

    n = metric.window_size

    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):

        if isinstance(yp, list):
            yp = dict(enumerate(yp))

        metric.update(yt, yp)

        if i >= 1:
            assert math.isclose(
                metric.get(),
                sk_metric(tail(y_true[:i + 1], n), tail(y_pred[:i + 1], n))
            )
