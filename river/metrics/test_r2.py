import collections
import math

from sklearn import metrics as sk_metrics

from river import metrics, utils


def test_r2():

    r2 = metrics.R2()
    sk_r2 = sk_metrics.r2_score
    y_true = [
        0.8454795371447003,
        0.36530165758399,
        0.32733508302313696,
        0.3907841858998481,
        0.33367434897950754,
        0.10209784710790504,
        0.9537676025825098,
        0.49208175447064406,
        0.25808584318657635,
        0.22114819033795075,
    ]
    y_pred = [
        0.28023834604821274,
        0.8799362767074241,
        0.08515114818265701,
        0.04474250926418322,
        0.34180002419963607,
        0.7018106760663595,
        0.4650385019574035,
        0.8556417963590652,
        0.6818470809869084,
        0.9232617479260311,
    ]
    weights = [
        0.8977831327937194,
        0.9059323375861669,
        0.6403106244128447,
        8.703927525188782e-05,
        0.6043234651744177,
        0.09393312409759613,
        0.24795625986595893,
        0.28872232042874824,
        0.6618185762206685,
        0.14885033958068794,
    ]

    for i, (yt, yp, w) in enumerate(zip(y_true, y_pred, weights)):

        r2.update(yt, yp, w)

        if i >= 1:
            assert math.isclose(
                r2.get(),
                sk_r2(y_true[: i + 1], y_pred[: i + 1], sample_weight=weights[: i + 1]),
            )


def test_rolling_r2():
    def tail(iterable, n):
        return collections.deque(iterable, maxlen=n)

    r2 = utils.Rolling(metrics.R2(), window_size=3)
    n = r2.window_size
    sk_r2 = sk_metrics.r2_score
    y_true = [
        0.4656520648923188,
        0.5768996330715701,
        0.045385529424484594,
        0.31852843450357393,
        0.8344133739124894,
    ]
    y_pred = [
        0.5431172475992199,
        0.2436885541729249,
        0.20238076597257637,
        0.6173775443360237,
        0.9194776501054074,
    ]

    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):

        r2.update(yt, yp)

        if i >= 2:
            assert math.isclose(r2.get(), sk_r2(tail(y_true[: i + 1], n), tail(y_pred[: i + 1], n)))
