from __future__ import annotations

import collections
import copy
import functools
import importlib
import inspect
import pickle
import platform
import random

import numpy as np
import pytest
from sklearn import metrics as sk_metrics

from river import metrics, utils


def load_metrics():
    """Yields all the metrics."""

    for name, obj in inspect.getmembers(importlib.import_module("river.metrics"), inspect.isclass):
        if name == "Metrics":
            continue

        if inspect.isabstract(obj):
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


@pytest.mark.parametrize(
    "metric",
    [pytest.param(metric, id=type(metric).__name__) for metric in load_metrics()],
)
def test_repr(metric):
    repr(metric)

    for y_true, y_pred, _ in generate_test_cases(metric=metric, n=10):
        m = copy.deepcopy(metric)

        for yt, yp in zip(y_true, y_pred):
            m.update(yt, yp)
            repr(metric)


def generate_test_cases(metric, n):
    """Yields n (y_true, y_pred, sample_weight) triplets for a given metric."""

    sample_weights = [random.random() for _ in range(n)]

    if isinstance(metric, metrics.base.ClassificationMetric):
        y_true = [random.choice([False, True]) for _ in range(n)]
        if metric.requires_labels:
            y_pred = [random.choice([False, True]) for _ in range(n)]
        else:
            y_pred = [dict(zip([False, True], np.random.dirichlet(np.ones(2)))) for _ in range(n)]
        yield y_true, y_pred, sample_weights

    if isinstance(metric, metrics.base.MultiClassMetric):
        y_true = [random.choice([0, 1, 2]) for _ in range(n)]
        if metric.requires_labels:
            y_pred = [random.choice([0, 1, 2]) for _ in range(n)]
        else:
            y_pred = [dict(zip([0, 1, 2], np.random.dirichlet(np.ones(3)))) for _ in range(n)]
        yield y_true, y_pred, sample_weights

    if isinstance(metric, metrics.base.RegressionMetric):
        yield (
            [random.random() for _ in range(n)],
            [random.random() for _ in range(n)],
            sample_weights,
        )


def partial(f, **kwargs):
    return functools.update_wrapper(functools.partial(f, **kwargs), f)


def roc_auc_score(y_true, y_score):
    """
    This functions is a wrapper to the scikit-learn roc_auc_score function.
    It was created because the scikit version utilizes array of scores and
    may raise a ValueError if there is only one class present in y_true.
    This wrapper returns 0 if y_true has only one class and
    deals with the scores.
    """
    nonzero = np.count_nonzero(y_true)
    if nonzero == 0 or nonzero == len(y_true):
        return 0

    scores = [s[True] for s in y_score]

    return sk_metrics.roc_auc_score(y_true, scores)


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
    (metrics.Homogeneity(), sk_metrics.homogeneity_score),
    (metrics.Completeness(), sk_metrics.completeness_score),
    (metrics.VBeta(beta=0.5), partial(sk_metrics.v_measure_score, beta=0.5)),
    (metrics.FowlkesMallows(), sk_metrics.fowlkes_mallows_score),
    (metrics.Rand(), sk_metrics.rand_score),
    (metrics.AdjustedRand(), sk_metrics.adjusted_rand_score),
    (metrics.MutualInfo(), sk_metrics.mutual_info_score),
    (
        metrics.NormalizedMutualInfo(average_method="min"),
        partial(sk_metrics.normalized_mutual_info_score, average_method="min"),
    ),
    (
        metrics.NormalizedMutualInfo(average_method="max"),
        partial(sk_metrics.normalized_mutual_info_score, average_method="max"),
    ),
    (
        metrics.NormalizedMutualInfo(average_method="arithmetic"),
        partial(sk_metrics.normalized_mutual_info_score, average_method="arithmetic"),
    ),
    (
        metrics.NormalizedMutualInfo(average_method="geometric"),
        partial(sk_metrics.normalized_mutual_info_score, average_method="geometric"),
    ),
    (
        metrics.AdjustedMutualInfo(average_method="max"),
        partial(sk_metrics.adjusted_mutual_info_score, average_method="max"),
    ),
    (
        metrics.AdjustedMutualInfo(average_method="arithmetic"),
        partial(sk_metrics.adjusted_mutual_info_score, average_method="arithmetic"),
    ),
    (
        metrics.AdjustedMutualInfo(average_method="geometric"),
        partial(sk_metrics.adjusted_mutual_info_score, average_method="geometric"),
    ),
    (metrics.Jaccard(), partial(sk_metrics.jaccard_score, average="binary")),
    (metrics.MacroJaccard(), partial(sk_metrics.jaccard_score, average="macro")),
    (metrics.MicroJaccard(), partial(sk_metrics.jaccard_score, average="micro")),
    (metrics.WeightedJaccard(), partial(sk_metrics.jaccard_score, average="weighted")),
    (metrics.RollingROCAUC(), roc_auc_score),
]

# HACK: not sure why this is needed, see this CI run https://github.com/online-ml/river/runs/7992357532?check_suite_focus=true
if platform.system() != "Linux":
    TEST_CASES.append(
        (
            metrics.AdjustedMutualInfo(average_method="min"),
            partial(sk_metrics.adjusted_mutual_info_score, average_method="min"),
        )
    )


@pytest.mark.parametrize(
    "metric, sk_metric",
    [
        pytest.param(metric, sk_metric, id=f"{metric.__class__.__name__}")
        for metric, sk_metric in TEST_CASES
    ],
)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning")
def test_metric(metric, sk_metric):
    for y_true, y_pred, sample_weights in generate_test_cases(metric=metric, n=30):
        m = copy.deepcopy(metric)
        for i, (yt, yp, w) in enumerate(zip(y_true, y_pred, sample_weights)):
            if metric.works_with_weights:
                m.update(y_true=yt, y_pred=yp, sample_weight=w)
            else:
                m.update(y_true=yt, y_pred=yp)

            if i >= 1:
                if metric.works_with_weights:
                    kwargs = {"sample_weight": sample_weights[: i + 1]}
                else:
                    kwargs = {}
                assert abs(m.get() - sk_metric(y_true[: i + 1], y_pred[: i + 1], **kwargs)) < 1e-6


@pytest.mark.parametrize(
    "metric, sk_metric",
    [
        pytest.param(metric, sk_metric, id=f"{metric.__class__.__name__}")
        for metric, sk_metric in TEST_CASES
    ],
)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning")
def test_rolling_metric(metric, sk_metric):
    def tail(iterable, n):
        return collections.deque(iterable, maxlen=n)

    for n in (1, 2, 5, 10):
        for y_true, y_pred, _ in generate_test_cases(metric=metric, n=30):
            m = utils.Rolling(copy.deepcopy(metric), window_size=n)

            # Check str works
            str(m)

            for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
                m.update(y_true=yt, y_pred=yp)

                if i >= 1:
                    assert (
                        abs(
                            m.get()
                            - sk_metric(
                                tail(y_true[: i + 1], n),
                                tail(y_pred[: i + 1], n),
                            )
                        )
                        < 1e-10
                    )


def test_compose():
    metrics.MAE() + metrics.MSE()
    metrics.Accuracy() + metrics.LogLoss()

    with pytest.raises(ValueError):
        _ = metrics.MSE() + metrics.LogLoss()

    with pytest.raises(ValueError):
        _ = metrics.MSE() + metrics.MAE() + metrics.LogLoss()
