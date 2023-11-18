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


def test_confusion_and_other_metrics():
    """

    >>> dataset = datasets.Phishing()

    >>> model = preprocessing.StandardScaler() | linear_model.LogisticRegression(
    ...     optimizer=optim.SGD(0.1)
    ... )

    >>> metric = metrics.ConfusionMatrix() + metrics.F1() + metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
            False   True
    False     613     89
     True      49    499
    F1: 87.85%
    Accuracy: 88.96%

    """
