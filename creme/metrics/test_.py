import collections
import copy
import functools
import importlib
import inspect
import math
import pickle
import random

from creme import metrics
from creme.metrics import base
import numpy as np
import pytest
from sklearn import metrics as sk_metrics


def load_metrics():
    """Yields all the metrics."""

    for name, obj in inspect.getmembers(importlib.import_module('creme.metrics'), inspect.isclass):

        if issubclass(obj, metrics.PerClass):
            yield obj(metric=metrics.Precision())
            continue

        elif issubclass(obj, metrics.Rolling):
            yield obj(metric=metrics.MSE(), window_size=42)
            continue

        elif name == 'RegressionMultiOutput':
            yield obj(metric=metrics.MSE())
            continue

        try:
            sig = inspect.signature(obj)
            yield obj(**{
                param.name: param.default if param.default != param.empty else 5
                for param in sig.parameters.values()
            })
        except ValueError:
            yield obj()


@pytest.mark.parametrize('metric', load_metrics(), ids=lambda metric: type(metric).__name__)
def test_pickling(metric):
    print(metric.__class__)
    assert isinstance(pickle.loads(pickle.dumps(metric)), metric.__class__)
    assert isinstance(copy.deepcopy(metric), metric.__class__)


def generate_test_cases(metric, n):
    """Yields (y_true, y_pred) pairs of size n for a given metric."""

    sample_weights = [random.random() for _ in range(n)]

    if isinstance(metric, base.BinaryMetric):
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
            sample_weights
        )


TEST_CASES = [
    (metrics.Accuracy(), sk_metrics.accuracy_score),
    (metrics.Precision(), sk_metrics.precision_score),
    (metrics.MacroPrecision(), functools.partial(sk_metrics.precision_score, average='macro')),
    (metrics.MicroPrecision(), functools.partial(sk_metrics.precision_score, average='micro')),
    (metrics.WeightedPrecision(), functools.partial(sk_metrics.precision_score, average='weighted')),
    (metrics.Recall(), sk_metrics.recall_score),
    (metrics.MacroRecall(), functools.partial(sk_metrics.recall_score, average='macro')),
    (metrics.MicroRecall(), functools.partial(sk_metrics.recall_score, average='micro')),
    (metrics.WeightedRecall(), functools.partial(sk_metrics.recall_score, average='weighted')),
    (metrics.FBeta(beta=.5), functools.partial(sk_metrics.fbeta_score, beta=.5)),
    (metrics.MacroFBeta(beta=.5), functools.partial(sk_metrics.fbeta_score, beta=.5, average='macro')),
    (metrics.MicroFBeta(beta=.5), functools.partial(sk_metrics.fbeta_score, beta=.5, average='micro')),
    (metrics.WeightedFBeta(beta=.5), functools.partial(sk_metrics.fbeta_score, beta=.5, average='weighted')),
    (metrics.F1(), sk_metrics.f1_score),
    (metrics.MacroF1(), functools.partial(sk_metrics.f1_score, average='macro')),
    (metrics.MicroF1(), functools.partial(sk_metrics.f1_score, average='micro')),
    (metrics.WeightedF1(), functools.partial(sk_metrics.f1_score, average='weighted')),
    (metrics.MCC(), sk_metrics.matthews_corrcoef),
    (metrics.MAE(), sk_metrics.mean_absolute_error),
    (metrics.MSE(), sk_metrics.mean_squared_error),
]


@pytest.mark.parametrize('metric, sk_metric', TEST_CASES)
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.filterwarnings('ignore::sklearn.metrics.classification.UndefinedMetricWarning')
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
                assert abs(
                    m.get() -
                    sk_metric(
                        y_true=y_true[:i + 1],
                        y_pred=y_pred[:i + 1],
                        sample_weight=sample_weights[:i + 1]
                    )
                ) < 1e-10


@pytest.mark.parametrize('metric, sk_metric', TEST_CASES)
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.filterwarnings('ignore::sklearn.metrics.classification.UndefinedMetricWarning')
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
                    assert abs(
                        m.get() -
                        sk_metric(
                            y_true=tail(y_true[:i + 1], n),
                            y_pred=tail(y_pred[:i + 1], n)
                        )
                    ) < 1e-10


def test_log_loss():

    metric = metrics.LogLoss()

    y_true = [True, False, False, True]
    y_pred = [0.9, 0.1, 0.2, 0.65]

    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        metric.update(yt, yp)

        if i >= 1:
            assert math.isclose(metric.get(), sk_metrics.log_loss(y_true[:i + 1], y_pred[:i + 1]))

    metric.revert(y_true[-1], y_pred[-1])
    assert math.isclose(metric.get(), sk_metrics.log_loss(y_true[:-1], y_pred[:-1]))


def test_cross_entropy():

    metric = metrics.CrossEntropy()

    y_true = [0, 1, 2, 2]
    y_pred = [
        [0.29450637, 0.34216758, 0.36332605],
        [0.21290077, 0.32728332, 0.45981591],
        [0.42860913, 0.33380113, 0.23758974],
        [0.44941979, 0.32962558, 0.22095463]
    ]

    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        yp = dict(enumerate(yp))
        metric.update(yt, yp)

        if i >= 1:
            assert math.isclose(
                metric.get(),
                sk_metrics.log_loss(y_true[:i + 1], y_pred[:i + 1], labels=[0, 1, 2])
            )

    metric.revert(y_true[-1], dict(enumerate(y_pred[-1])))
    assert math.isclose(metric.get(), sk_metrics.log_loss(y_true[:-1], y_pred[:-1]))


@pytest.mark.filterwarnings('ignore::sklearn.metrics.classification.UndefinedMetricWarning')
def test_multi_fbeta():

    fbeta = metrics.MultiFBeta(betas={0: 0.25, 1: 1, 2: 4}, weights={0: 1, 1: 1, 2: 2})
    sk_fbeta = sk_metrics.fbeta_score
    y_true = [0, 1, 2, 2, 2]
    y_pred = [0, 1, 0, 2, 1]

    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):

        fbeta.update(yt, yp)

        if i >= 2:
            fbeta_0, _, _ = sk_fbeta(y_true[:i + 1], y_pred[:i + 1], beta=0.25, average=None)
            _, fbeta_1, _ = sk_fbeta(y_true[:i + 1], y_pred[:i + 1], beta=1, average=None)
            _, _, fbeta_2 = sk_fbeta(y_true[:i + 1], y_pred[:i + 1], beta=4, average=None)

            multi_fbeta = fbeta_0 * 1 + fbeta_1 * 1 + fbeta_2 * 2
            multi_fbeta /= (1 + 1 + 2)

            assert math.isclose(fbeta.get(), multi_fbeta)


@pytest.mark.filterwarnings('ignore::sklearn.metrics.classification.UndefinedMetricWarning')
def test_rolling_multi_f1():

    def tail(iterable, n):
        return collections.deque(iterable, maxlen=n)

    fbeta = metrics.Rolling(
        metric=metrics.MultiFBeta(
            betas={0: 0.25, 1: 1, 2: 4},
            weights={0: 1, 1: 1, 2: 2}
        ),
        window_size=3
    )
    n = fbeta.window_size
    sk_fbeta = sk_metrics.fbeta_score
    y_true = [0, 1, 2, 2, 2]
    y_pred = [0, 1, 0, 2, 1]

    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):

        fbeta.update(yt, yp)

        if i >= 2:
            sk_y_true, sk_y_pred = tail(y_true[:i + 1], n), tail(y_pred[:i + 1], n)
            fbeta_0, _, _ = sk_fbeta(sk_y_true, sk_y_pred, beta=0.25, average=None)
            _, fbeta_1, _ = sk_fbeta(sk_y_true, sk_y_pred, beta=1, average=None)
            _, _, fbeta_2 = sk_fbeta(sk_y_true, sk_y_pred, beta=4, average=None)

            multi_fbeta = fbeta_0 * 1 + fbeta_1 * 1 + fbeta_2 * 2
            multi_fbeta /= (1 + 1 + 2)

            assert math.isclose(fbeta.get(), multi_fbeta)


def test_compose():

    with pytest.raises(ValueError):
        _ = metrics.MSE() + metrics.LogLoss()

    with pytest.raises(ValueError):
        _ = metrics.MSE() + metrics.MAE() + metrics.LogLoss()
