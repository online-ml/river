"""Plain estimator run protocols for the regression suite.

Every harness accepts an estimator plus materialized samples and returns a
``RunResult``. Metrics are gated by tolerances in ``compare.py``. Diagnostics
are written to the YAML artifact for debugging, but are never gated.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any, NamedTuple

from benchmarks.estimator_regression import workloads
from benchmarks.estimator_regression.scenarios import Scenario
from river import metrics

MetricDict = dict[str, float]
Harness = Callable[[Any, list[Any]], "RunResult"]


class RunResult(NamedTuple):
    """Metrics that gate CI plus informational diagnostics."""

    metrics: MetricDict
    diagnostics: MetricDict


_BINARY_METRICS = ("accuracy", "f1")
_MULTICLASS_METRICS = ("accuracy", "macro_f1", "micro_f1")
_REGRESSION_METRICS = ("mae", "rmse", "r2")
_MULTILABEL_METRICS = ("exact_match", "macro_accuracy")
_MULTITARGET_METRICS = ("multioutput_mae", "multioutput_rmse")
_METRIC_BUILDERS: dict[str, Callable[[], metrics.base.Metric]] = {
    "accuracy": metrics.Accuracy,
    "f1": metrics.F1,
    "macro_f1": metrics.MacroF1,
    "micro_f1": metrics.MicroF1,
    "log_loss": metrics.LogLoss,
    "cross_entropy": metrics.CrossEntropy,
    "mae": metrics.MAE,
    "rmse": metrics.RMSE,
    "r2": metrics.R2,
    "exact_match": metrics.multioutput.ExactMatch,
    "macro_accuracy": lambda: metrics.multioutput.MacroAverage(metrics.Accuracy()),
    "multioutput_mae": lambda: metrics.multioutput.PerOutput(metrics.MAE()),
    "multioutput_rmse": lambda: metrics.multioutput.PerOutput(metrics.RMSE()),
}


def run_scenario(scenario: Scenario) -> RunResult:
    """Build an estimator, load its workload, and run the matching harness."""

    estimator = scenario.build()
    make_samples = getattr(workloads, scenario.workload)
    samples = make_samples(scenario.n_samples)
    return HARNESSES[scenario.harness](estimator, samples)


def _round(value: float) -> float:
    """Round metrics so YAML artifacts stay byte-stable across platforms."""

    return float(round(float(value), 12))


def _new_metric(name: str) -> metrics.base.Metric:
    """Create a fresh River metric instance by artifact metric name."""

    return _METRIC_BUILDERS[name]()


def _final_metric_value(metric: metrics.base.Metric) -> float:
    """Return a scalar from a River metric.

    ``PerOutput`` metrics return a dict of child metrics from ``get()``. The
    regression suite stores one scalar per metric, so multi-output metrics are
    averaged into a single comparable value.
    """

    value = metric.get()
    if isinstance(value, dict):
        values = [_final_metric_value(child_metric) for child_metric in value.values()]
        return float(sum(values) / max(len(values), 1))
    return float(value)


def _probabilities(estimator: Any, x: dict[str, Any]) -> dict[Any, float] | None:
    """Return class probabilities when the estimator implements them.

    River classifiers may expose ``predict_proba_one`` but raise
    ``NotImplementedError`` by default.
    """

    if not hasattr(estimator, "predict_proba_one"):
        return None
    try:
        probabilities = estimator.predict_proba_one(x)
    except NotImplementedError:
        return None
    return probabilities or None


def _run_supervised(
    estimator: Any,
    samples: list[Any],
    metric_names: tuple[str, ...],
    *,
    proba_metric: str | None = None,
) -> RunResult:
    metric_instances = {name: _new_metric(name) for name in metric_names}
    proba_instance = _new_metric(proba_metric) if proba_metric else None
    n_predictions = 0
    saw_probabilities = False

    for x, y in samples:
        prediction = estimator.predict_one(x)
        if prediction is not None:
            n_predictions += 1
            for metric in metric_instances.values():
                metric.update(y, prediction)
        if proba_instance is not None:
            probabilities = _probabilities(estimator, x)
            if probabilities:
                saw_probabilities = True
                proba_instance.update(y, probabilities)
        estimator.learn_one(x, y)

    values = {
        name: _round(_final_metric_value(metric)) for name, metric in metric_instances.items()
    }
    if proba_instance is not None and saw_probabilities:
        assert proba_metric is not None
        values[proba_metric] = _round(_final_metric_value(proba_instance))
    return RunResult(
        values,
        {"n_predictions": float(n_predictions), "n_learned": float(len(samples))},
    )


def binary_classification(estimator: Any, samples: list[Any]) -> RunResult:
    """Run predict-then-learn validation for binary classifiers."""

    return _run_supervised(estimator, samples, _BINARY_METRICS, proba_metric="log_loss")


def multiclass_classification(estimator: Any, samples: list[Any]) -> RunResult:
    """Run predict-then-learn validation for multiclass classifiers."""

    return _run_supervised(
        estimator,
        samples,
        _MULTICLASS_METRICS,
        proba_metric="cross_entropy",
    )


def regression(estimator: Any, samples: list[Any]) -> RunResult:
    """Run predict-then-learn validation for regressors."""

    return _run_supervised(estimator, samples, _REGRESSION_METRICS)


def _run_dict_target(
    estimator: Any, samples: list[Any], metric_names: tuple[str, ...]
) -> RunResult:
    metric_instances = {name: _new_metric(name) for name in metric_names}
    n_predictions = 0

    for x, y in samples:
        prediction = estimator.predict_one(x)
        if prediction:
            n_predictions += 1
            for metric in metric_instances.values():
                metric.update(y, prediction)
        estimator.learn_one(x, y)

    return RunResult(
        {name: _round(_final_metric_value(metric)) for name, metric in metric_instances.items()},
        {"n_predictions": float(n_predictions), "n_learned": float(len(samples))},
    )


def multilabel_classification(estimator: Any, samples: list[Any]) -> RunResult:
    """Run predict-then-learn validation for multi-label classifiers."""

    return _run_dict_target(estimator, samples, _MULTILABEL_METRICS)


def multitarget_regression(estimator: Any, samples: list[Any]) -> RunResult:
    """Run predict-then-learn validation for multi-target regressors."""

    return _run_dict_target(estimator, samples, _MULTITARGET_METRICS)


def anomaly(estimator: Any, samples: list[Any]) -> RunResult:
    """Score-then-learn validation for anomaly detectors."""

    roc_auc = metrics.RollingROCAUC(window_size=10_000)
    anomaly_scores: list[float] = []
    normal_scores: list[float] = []

    for x, y in samples:
        score = estimator.score_one(x)
        estimator.learn_one(x)
        if score is None:
            continue
        score = float(score)
        roc_auc.update(bool(y), score)
        if bool(y):
            anomaly_scores.append(score)
        else:
            normal_scores.append(score)

    mean_anomaly = sum(anomaly_scores) / len(anomaly_scores) if anomaly_scores else 0.0
    mean_normal = sum(normal_scores) / len(normal_scores) if normal_scores else 0.0
    return RunResult(
        {
            "rolling_roc_auc": _round(float(roc_auc.get())),
            "score_separation": _round(mean_anomaly - mean_normal),
        },
        {"n_scored": float(len(anomaly_scores) + len(normal_scores))},
    )


def clustering(estimator: Any, samples: list[Any]) -> RunResult:
    """Learn-then-predict validation for clusterers."""

    silhouette = metrics.Silhouette()
    predicted_clusters: set[int] = set()
    n_predictions = 0

    for x in samples:
        estimator.learn_one(x)
        prediction = estimator.predict_one(x)
        if prediction is None:
            continue
        n_predictions += 1
        predicted_clusters.add(prediction)
        centers = getattr(estimator, "centers", None)
        if centers:
            silhouette.update(x, prediction, centers)

    return RunResult(
        {"n_clusters": float(len(predicted_clusters)), "n_predictions": float(n_predictions)},
        {"silhouette": _round(float(silhouette.get()))},
    )


def _forecast_one(estimator: Any, x: dict[str, float] | None) -> float | None:
    try:
        forecast = estimator.forecast(horizon=1, xs=[x] if x else None)
    except (IndexError, ValueError):
        return None
    if not forecast:
        return None
    return float(forecast[0])


def forecasting(estimator: Any, samples: list[Any]) -> RunResult:
    """One-step-ahead validation for time-series forecasters."""

    metric_instances = {name: _new_metric(name) for name in ("mae", "rmse")}
    n_predictions = 0

    for x, y in samples:
        prediction = _forecast_one(estimator, x)
        if prediction is not None:
            n_predictions += 1
            for metric in metric_instances.values():
                metric.update(y, prediction)
        estimator.learn_one(y=y, x=x)

    return RunResult(
        {name: _round(_final_metric_value(metric)) for name, metric in metric_instances.items()},
        {"n_predictions": float(n_predictions)},
    )


def recommendation(estimator: Any, samples: list[Any]) -> RunResult:
    """Predict-then-learn validation for recommenders."""

    absolute_errors: list[float] = []
    hits = 0
    n_ranked = 0
    all_items = {item for _user, item, _rating in samples}

    for user, item, rating in samples:
        prediction = estimator.predict_one(user, item)
        if prediction is not None:
            absolute_errors.append(abs(float(prediction) - float(rating)))
        ranking = estimator.rank(user, all_items)
        if ranking:
            n_ranked += 1
            if ranking[0] == item:
                hits += 1
        estimator.learn_one(user, item, rating)

    mae = sum(absolute_errors) / len(absolute_errors) if absolute_errors else 0.0
    hit_rate = hits / n_ranked if n_ranked else 0.0
    return RunResult(
        {"mae": _round(mae)},
        {"top1_hit_rate": _round(hit_rate), "n_predictions": float(len(absolute_errors))},
    )


def behavioral_invariant(estimator: Any, samples: list[Any]) -> RunResult:
    """Run learn/predict and record stable behavioral invariants."""

    n_predictions = 0
    predictions: list[str] = []
    hasher = hashlib.sha256()

    for x, y in samples:
        prediction = estimator.predict_one(x) if hasattr(estimator, "predict_one") else None
        if prediction is not None:
            n_predictions += 1
            prediction_text = str(prediction)
            predictions.append(prediction_text)
            hasher.update(prediction_text.encode("utf-8"))
        if hasattr(estimator, "learn_one"):
            estimator.learn_one(x, y)

    # Keep the digest numeric so it fits the same YAML shape as other results.
    digest = float(int(hasher.hexdigest(), 16) % 1_000_000) / 1_000_000.0
    return RunResult(
        {
            "n_predictions": float(n_predictions),
            "n_unique_predictions": float(len(set(predictions))),
        },
        {"prediction_digest": digest},
    )


HARNESSES: dict[str, Harness] = {
    "anomaly": anomaly,
    "behavioral_invariant": behavioral_invariant,
    "binary_classification": binary_classification,
    "clustering": clustering,
    "forecasting": forecasting,
    "multiclass_classification": multiclass_classification,
    "multilabel_classification": multilabel_classification,
    "multitarget_regression": multitarget_regression,
    "recommendation": recommendation,
    "regression": regression,
}
