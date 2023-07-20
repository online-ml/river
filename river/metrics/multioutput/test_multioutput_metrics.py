from __future__ import annotations

import collections
import functools
import math
import random

import pandas as pd
import pytest
from sklearn import metrics as sk_metrics

from river import metrics


@pytest.mark.parametrize(
    "metric, sk_metric",
    [
        pytest.param(
            metric,
            sk_metric,
            id=f"{metric.__class__.__name__}"
            + (f"({metric.metric.__class__.__name__})" if hasattr(metric, "metric") else ""),
        )
        for metric, sk_metric in [
            (
                metrics.multioutput.ExactMatch(),
                sk_metrics.accuracy_score,
            ),
            (
                metrics.multioutput.MacroAverage(metrics.Precision()),
                functools.partial(sk_metrics.precision_score, average="macro", zero_division=0),
            ),
            (
                metrics.multioutput.MicroAverage(metrics.Precision()),
                functools.partial(sk_metrics.precision_score, average="micro", zero_division=0),
            ),
            (
                metrics.multioutput.SampleAverage(metrics.Precision()),
                functools.partial(sk_metrics.precision_score, average="samples", zero_division=0),
            ),
            (
                metrics.multioutput.MacroAverage(metrics.Recall()),
                functools.partial(sk_metrics.recall_score, average="macro", zero_division=0),
            ),
            (
                metrics.multioutput.MicroAverage(metrics.Recall()),
                functools.partial(sk_metrics.recall_score, average="micro", zero_division=0),
            ),
            (
                metrics.multioutput.SampleAverage(metrics.Recall()),
                functools.partial(sk_metrics.recall_score, average="samples", zero_division=0),
            ),
            (
                metrics.multioutput.MacroAverage(metrics.F1()),
                functools.partial(sk_metrics.f1_score, average="macro", zero_division=0),
            ),
            (
                metrics.multioutput.MicroAverage(metrics.F1()),
                functools.partial(sk_metrics.f1_score, average="micro", zero_division=0),
            ),
            (
                metrics.multioutput.SampleAverage(metrics.F1()),
                functools.partial(sk_metrics.f1_score, average="samples", zero_division=0),
            ),
        ]
    ],
)
def test_multiout_binary_clf(metric, sk_metric):
    y_true = []
    y_pred = []
    for _ in range(30):
        y_true.append({i: random.random() < 0.3 for i in range(3)})
        y_pred.append({i: random.random() < 0.3 for i in range(3)})
    y_true = pd.DataFrame(y_true)
    y_pred = pd.DataFrame(y_pred)

    for i, (yt, yp) in enumerate(
        zip(y_true.to_dict(orient="records"), y_pred.to_dict(orient="records"))
    ):
        metric.update(yt, yp)
        if i == 0:
            continue

        A = metric.get()
        B = sk_metric(y_true[: i + 1], y_pred[: i + 1])
        assert math.isclose(A, B, abs_tol=1e-3)


@pytest.mark.parametrize(
    "metric, sk_metric",
    [
        pytest.param(
            metric,
            sk_metric,
            id=f"{metric.__class__.__name__}"
            + (f"({metric.metric.__class__.__name__})" if hasattr(metric, "metric") else ""),
        )
        for metric, sk_metric in [
            # Both of the following cases are the same, because the average of averages is the same
            # as the global average
            (
                metrics.multioutput.MacroAverage(metrics.MAE()),
                functools.partial(sk_metrics.mean_absolute_error, multioutput="uniform_average"),
            ),
            (
                metrics.multioutput.MicroAverage(metrics.MAE()),
                functools.partial(sk_metrics.mean_absolute_error, multioutput="uniform_average"),
            ),
            #
            (
                metrics.multioutput.PerOutput(metrics.MAE()),
                functools.partial(sk_metrics.mean_absolute_error, multioutput="raw_values"),
            ),
        ]
    ],
)
def test_multiout_regression(metric, sk_metric):
    y_true = []
    y_pred = []
    for _ in range(30):
        y_true.append({i: random.random() for i in range(3)})
        y_pred.append({i: random.random() for i in range(3)})
    y_true = pd.DataFrame(y_true)
    y_pred = pd.DataFrame(y_pred)

    for i, (yt, yp) in enumerate(
        zip(y_true.to_dict(orient="records"), y_pred.to_dict(orient="records"))
    ):
        metric.update(yt, yp)
        if i == 0:
            continue

        A = metric.get()
        B = sk_metric(y_true[: i + 1], y_pred[: i + 1])

        if isinstance(A, collections.abc.Mapping):
            for k in A:
                assert math.isclose(A[k].get(), B[k], abs_tol=1e-3)
        else:
            assert math.isclose(A, B, abs_tol=1e-3)
