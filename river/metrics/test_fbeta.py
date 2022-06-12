import collections
import math

from sklearn import metrics as sk_metrics

from river import metrics, utils


def test_multi_fbeta():

    fbeta = metrics.MultiFBeta(betas={0: 0.25, 1: 1, 2: 4}, weights={0: 1, 1: 1, 2: 2})
    sk_fbeta = sk_metrics.fbeta_score
    y_true = [0, 1, 2, 2, 2]
    y_pred = [0, 1, 0, 2, 1]

    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):

        fbeta.update(yt, yp)

        if i >= 2:
            fbeta_0, _, _ = sk_fbeta(y_true[: i + 1], y_pred[: i + 1], beta=0.25, average=None)
            _, fbeta_1, _ = sk_fbeta(y_true[: i + 1], y_pred[: i + 1], beta=1, average=None)
            _, _, fbeta_2 = sk_fbeta(y_true[: i + 1], y_pred[: i + 1], beta=4, average=None)

            multi_fbeta = fbeta_0 * 1 + fbeta_1 * 1 + fbeta_2 * 2
            multi_fbeta /= 1 + 1 + 2

            assert math.isclose(fbeta.get(), multi_fbeta)


def test_rolling_multi_fbeta():
    def tail(iterable, n):
        return collections.deque(iterable, maxlen=n)

    fbeta = utils.Rolling(
        metrics.MultiFBeta(betas={0: 0.25, 1: 1, 2: 4}, weights={0: 1, 1: 1, 2: 2}),
        window_size=3,
    )
    n = fbeta.window_size
    sk_fbeta = sk_metrics.fbeta_score
    y_true = [0, 1, 2, 2, 2]
    y_pred = [0, 1, 0, 2, 1]

    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):

        fbeta.update(yt, yp)

        if i >= 2:
            sk_y_true, sk_y_pred = tail(y_true[: i + 1], n), tail(y_pred[: i + 1], n)
            fbeta_0, _, _ = sk_fbeta(sk_y_true, sk_y_pred, beta=0.25, average=None)
            _, fbeta_1, _ = sk_fbeta(sk_y_true, sk_y_pred, beta=1, average=None)
            _, _, fbeta_2 = sk_fbeta(sk_y_true, sk_y_pred, beta=4, average=None)

            multi_fbeta = fbeta_0 * 1 + fbeta_1 * 1 + fbeta_2 * 2
            multi_fbeta /= 1 + 1 + 2

            assert math.isclose(fbeta.get(), multi_fbeta)
