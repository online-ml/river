import math
import random
import pandas as pd
import pytest
from river import metrics
from sklearn import metrics as sk_metrics

TEST_CASES = [
    (
        metrics.multioutput.ExactMatch(),
        sk_metrics.accuracy_score,
    ),
]


@pytest.mark.parametrize(
    "metric, sk_metric",
    [
        pytest.param(metric, sk_metric, id=f"{metric.__class__.__name__}")
        for metric, sk_metric in TEST_CASES
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

        assert math.isclose(sk_metric(y_true[: i + 1], y_pred[: i + 1]), metric.get(), abs_tol=1e-3)
