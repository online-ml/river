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


def pr_auc_score(y_true, y_score):
    """
    This function is a wrapper to the scikit-learn precision_recall_curve and
    auc functions. Returns 0 if y_true has only one class.
    """
    nonzero = np.count_nonzero(y_true)
    if nonzero == 0 or nonzero == len(y_true):
        return 0

    scores = [s[True] for s in y_score]
    precision, recall, _ = sk_metrics.precision_recall_curve(y_true, scores)

    # Monotonic. decreasing
    precision = np.maximum.accumulate(precision)

    return sk_metrics.auc(recall, precision)


TEST_CASES = [
    (metrics.Accuracy(), sk_metrics.accuracy_score),
    (metrics.BalancedAccuracy(), sk_metrics.balanced_accuracy_score),
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
    (metrics.RollingPRAUC(), pr_auc_score),
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
                m.update(y_true=yt, y_pred=yp, w=w)
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
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
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

                if i >= 2:
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


def test_metrics_collection_forwards_sample_weight():
    # A Metrics collection must forward the sample weight to its children, just
    # like a standalone metric does (and like Metrics.revert already does).
    coll = metrics.MAE() + metrics.MSE()
    standalone = metrics.MAE()
    for y_true, y_pred, w in [(0, 0, 1.0), (0, 10, 99.0)]:
        coll.update(y_true, y_pred, w)
        standalone.update(y_true, y_pred, w)
    assert coll[0].get() == standalone.get() == pytest.approx(9.9)

    # update(w) and revert(w) must cancel exactly, leaving empty-state values.
    coll.revert(0, 0, 1.0)
    coll.revert(0, 10, 99.0)
    assert coll[0].get() == pytest.approx(0.0)


def test_cohen_kappa_supports_sample_weight():
    # CohenKappa advertises works_with_weights (it does not override the default), so its
    # score must honour the sample weight, exactly like sklearn.metrics.cohen_kappa_score.
    # The observed agreement and the expected-agreement terms have to be divided by the
    # total weight, not by the raw number of observations (which is what Accuracy does).
    y_true = ["cat", "ant", "cat", "cat", "ant", "bird", "cat", "ant"]
    y_pred = ["ant", "ant", "cat", "cat", "ant", "cat", "cat", "bird"]
    weights = [1.0, 5.0, 1.0, 3.0, 1.0, 2.0, 4.0, 1.0]

    metric = metrics.CohenKappa()
    for yt, yp, w in zip(y_true, y_pred, weights):
        metric.update(yt, yp, w)

    assert metric.get() == pytest.approx(
        sk_metrics.cohen_kappa_score(y_true, y_pred, sample_weight=weights)
    )

    # The weighted score must differ from the unweighted one, proving the weight is used.
    unweighted = metrics.CohenKappa()
    for yt, yp in zip(y_true, y_pred):
        unweighted.update(yt, yp)
    assert metric.get() != pytest.approx(unweighted.get())


def test_balanced_accuracy_ignores_unseen_predicted_classes():
    # A class that only shows up in the predictions has no support, so its recall is
    # undefined and must be excluded from the average, exactly like
    # sklearn.metrics.balanced_accuracy_score. Otherwise the score is deflated by
    # dividing over too many classes.
    y_true = [0, 0, 1, 1]
    y_pred = [0, 2, 1, 1]  # class 2 never appears as a true label

    metric = metrics.BalancedAccuracy()
    for yt, yp in zip(y_true, y_pred):
        metric.update(yt, yp)

    # recall(0) = 1/2, recall(1) = 2/2, class 2 dropped -> (0.5 + 1.0) / 2
    assert metric.get() == pytest.approx(sk_metrics.balanced_accuracy_score(y_true, y_pred))
    assert metric.get() == pytest.approx(0.75)
