import collections
import copy
import functools
import importlib
import inspect
import math
import pickle
import random

import numpy as np
import pytest
from sklearn import metrics as sk_metrics

from river import metrics
from river.metrics import base


def load_metrics():
    """Yields all the metrics."""

    for name, obj in inspect.getmembers(
        importlib.import_module("river.metrics"), inspect.isclass
    ):

        if name == "Metrics":
            continue

        if inspect.isabstract(obj):
            continue

        if issubclass(obj, metrics.Rolling):
            yield obj(metric=metrics.MSE(), window_size=42)
            continue

        elif name == "RegressionMultiOutput":
            yield obj(metric=metrics.MSE())
            continue

        try:
            sig = inspect.signature(obj)
            yield obj(
                **{
                    param.name: param.default if param.default != param.empty else 5
                    for param in sig.parameters.values()
                }
            )
        except ValueError:
            yield obj()


@pytest.mark.parametrize(
    "metric",
    [pytest.param(metric, id=type(metric).__name__) for metric in load_metrics()],
)
def test_pickling(metric):
    assert isinstance(pickle.loads(pickle.dumps(metric)), metric.__class__)
    assert isinstance(copy.deepcopy(metric), metric.__class__)


def generate_test_cases(metric, n):
    """Yields n (y_true, y_pred, sample_weight) triplets for a given metric."""

    sample_weights = [random.random() for _ in range(n)]

    if isinstance(metric, base.ClassificationMetric):
        y_true = [random.choice([False, True]) for _ in range(n)]
        if metric.requires_labels:
            y_pred = [random.choice([False, True]) for _ in range(n)]
        else:
            y_pred = [np.random.dirichlet(np.ones(2)).tolist() for _ in range(n)]
        yield y_true, y_pred, sample_weights

    if isinstance(metric, base.MultiClassMetric):
        y_true = [random.choice([0, 1, 2]) for _ in range(n)]
        if metric.requires_labels:
            y_pred = [random.choice([0, 1, 2]) for _ in range(n)]
        else:
            y_pred = [np.random.dirichlet(np.ones(3)).tolist() for _ in range(n)]
        yield y_true, y_pred, sample_weights

    if isinstance(metric, base.RegressionMetric):
        yield (
            [random.random() for _ in range(n)],
            [random.random() for _ in range(n)],
            sample_weights,
        )


def partial(f, **kwargs):
    return functools.update_wrapper(functools.partial(f, **kwargs), f)


TEST_CASES = [
    (metrics.Accuracy(), sk_metrics.accuracy_score),
    (metrics.Precision(), partial(sk_metrics.precision_score, zero_division=0)),
    (
        metrics.MacroPrecision(),
        partial(sk_metrics.precision_score, average="macro", zero_division=0),
    ),
    (
        metrics.MicroPrecision(),
        partial(sk_metrics.precision_score, average="micro", zero_division=0),
    ),
    (
        metrics.WeightedPrecision(),
        partial(sk_metrics.precision_score, average="weighted", zero_division=0),
    ),
    (metrics.Recall(), partial(sk_metrics.recall_score, zero_division=0)),
    (
        metrics.MacroRecall(),
        partial(sk_metrics.recall_score, average="macro", zero_division=0),
    ),
    (
        metrics.MicroRecall(),
        partial(sk_metrics.recall_score, average="micro", zero_division=0),
    ),
    (
        metrics.WeightedRecall(),
        partial(sk_metrics.recall_score, average="weighted", zero_division=0),
    ),
    (
        metrics.FBeta(beta=0.5),
        partial(sk_metrics.fbeta_score, beta=0.5, zero_division=0),
    ),
    (
        metrics.MacroFBeta(beta=0.5),
        partial(sk_metrics.fbeta_score, beta=0.5, average="macro", zero_division=0),
    ),
    (
        metrics.MicroFBeta(beta=0.5),
        partial(sk_metrics.fbeta_score, beta=0.5, average="micro", zero_division=0),
    ),
    (
        metrics.WeightedFBeta(beta=0.5),
        partial(sk_metrics.fbeta_score, beta=0.5, average="weighted", zero_division=0),
    ),
    (metrics.F1(), partial(sk_metrics.f1_score, zero_division=0)),
    (metrics.MacroF1(), partial(sk_metrics.f1_score, average="macro", zero_division=0)),
    (metrics.MicroF1(), partial(sk_metrics.f1_score, average="micro", zero_division=0)),
    (
        metrics.WeightedF1(),
        partial(sk_metrics.f1_score, average="weighted", zero_division=0),
    ),
    (metrics.MCC(), sk_metrics.matthews_corrcoef),
    (metrics.MAE(), sk_metrics.mean_absolute_error),
    (metrics.MSE(), sk_metrics.mean_squared_error),
]


@pytest.mark.parametrize(
    "metric, sk_metric",
    [
        pytest.param(metric, sk_metric, id=f"{metric.__class__.__name__}")
        for metric, sk_metric in TEST_CASES
    ],
)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_metric(metric, sk_metric):

    # Check str works
    str(metric)

    for y_true, y_pred, sample_weights in generate_test_cases(metric=metric, n=30):

        m = copy.deepcopy(metric)
        for i, (yt, yp, w) in enumerate(zip(y_true, y_pred, sample_weights)):

            if isinstance(yp, list):
                yp = dict(enumerate(yp))

            m.update(y_true=yt, y_pred=yp, sample_weight=w)

            if i >= 1:
                assert (
                    abs(
                        m.get()
                        - sk_metric(
                            y_true=y_true[: i + 1],
                            y_pred=y_pred[: i + 1],
                            sample_weight=sample_weights[: i + 1],
                        )
                    )
                    < 1e-6
                )


@pytest.mark.parametrize(
    "metric, sk_metric",
    [
        pytest.param(metric, sk_metric, id=f"{metric.__class__.__name__}")
        for metric, sk_metric in TEST_CASES
    ],
)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_rolling_metric(metric, sk_metric):
    def tail(iterable, n):
        return collections.deque(iterable, maxlen=n)

    for n in (1, 2, 5, 10):
        for y_true, y_pred, _ in generate_test_cases(metric=metric, n=30):

            m = metrics.Rolling(metric=copy.deepcopy(metric), window_size=n)

            # Check str works
            str(m)

            for i, (yt, yp) in enumerate(zip(y_true, y_pred)):

                if isinstance(yp, list):
                    yp = dict(enumerate(yp))

                m.update(y_true=yt, y_pred=yp)

                if i >= 1:
                    assert (
                        abs(
                            m.get()
                            - sk_metric(
                                y_true=tail(y_true[: i + 1], n),
                                y_pred=tail(y_pred[: i + 1], n),
                            )
                        )
                        < 1e-10
                    )


def test_log_loss():

    metric = metrics.LogLoss()

    y_true = [True, False, False, True]
    y_pred = [0.9, 0.1, 0.2, 0.65]

    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        metric.update(yt, yp)

        if i >= 1:
            assert math.isclose(
                metric.get(), sk_metrics.log_loss(y_true[: i + 1], y_pred[: i + 1])
            )

    metric.revert(y_true[-1], y_pred[-1])
    assert math.isclose(metric.get(), sk_metrics.log_loss(y_true[:-1], y_pred[:-1]))


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


def test_multi_fbeta():

    fbeta = metrics.MultiFBeta(betas={0: 0.25, 1: 1, 2: 4}, weights={0: 1, 1: 1, 2: 2})
    sk_fbeta = sk_metrics.fbeta_score
    y_true = [0, 1, 2, 2, 2]
    y_pred = [0, 1, 0, 2, 1]

    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):

        fbeta.update(yt, yp)

        if i >= 2:
            fbeta_0, _, _ = sk_fbeta(
                y_true[: i + 1], y_pred[: i + 1], beta=0.25, average=None
            )
            _, fbeta_1, _ = sk_fbeta(
                y_true[: i + 1], y_pred[: i + 1], beta=1, average=None
            )
            _, _, fbeta_2 = sk_fbeta(
                y_true[: i + 1], y_pred[: i + 1], beta=4, average=None
            )

            multi_fbeta = fbeta_0 * 1 + fbeta_1 * 1 + fbeta_2 * 2
            multi_fbeta /= 1 + 1 + 2

            assert math.isclose(fbeta.get(), multi_fbeta)


def test_rolling_multi_fbeta():
    def tail(iterable, n):
        return collections.deque(iterable, maxlen=n)

    fbeta = metrics.Rolling(
        metric=metrics.MultiFBeta(
            betas={0: 0.25, 1: 1, 2: 4}, weights={0: 1, 1: 1, 2: 2}
        ),
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

    r2 = metrics.Rolling(metric=metrics.R2(), window_size=3)
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
            assert math.isclose(
                r2.get(), sk_r2(tail(y_true[: i + 1], n), tail(y_pred[: i + 1], n))
            )


def test_compose():

    metrics.MAE() + metrics.MSE()
    metrics.Accuracy() + metrics.LogLoss()

    with pytest.raises(ValueError):
        _ = metrics.MSE() + metrics.LogLoss()

    with pytest.raises(ValueError):
        _ = metrics.MSE() + metrics.MAE() + metrics.LogLoss()
