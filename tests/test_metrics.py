from creme import metrics
from sklearn import metrics as sk_metrics


def test_log_loss():

    y_true = [True, False, False, True]
    y_pred = [0.9, 0.1, 0.2, 0.65]

    metric = metrics.LogLoss()
    for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
        metric = metric.update(y_t, y_p)
        if i >= 1:
            assert metric.get() == sk_metrics.log_loss(y_true[:i + 1], y_pred[:i + 1])


def test_cross_entropy():

    y_true = [0, 1, 2, 2]
    y_pred = [
        [0.29450637, 0.34216758, 0.36332605],
        [0.21290077, 0.32728332, 0.45981591],
        [0.42860913, 0.33380113, 0.23758974],
        [0.44941979, 0.32962558, 0.22095463]
    ]

    metric = metrics.CrossEntropy()
    labels = list(set(y_true))

    for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
        metric = metric.update(y_t, {i: p for i, p in enumerate(y_p)})
        if i >= 1:
            assert metric.get() == sk_metrics.log_loss(y_true[:i + 1], y_pred[:i + 1], labels=labels)
