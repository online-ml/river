from __future__ import annotations

from river import datasets, evaluate, linear_model, metrics, optim, preprocessing


def test_issue_1443():
    dataset = datasets.Phishing()

    model = preprocessing.StandardScaler() | linear_model.LogisticRegression(
        optimizer=optim.SGD(0.1)
    )

    metric = metrics.ConfusionMatrix()

    for _ in evaluate.iter_progressive_val_score(dataset, model, metric):
        pass
