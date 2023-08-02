from __future__ import annotations

import importlib
import inspect

import pytest

from river import (
    bandit,
    datasets,
    evaluate,
    linear_model,
    metrics,
    model_selection,
    optim,
    preprocessing,
)


def test_1259():
    """

    https://github.com/online-ml/river/issues/1259

    >>> from river import bandit
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import model_selection
    >>> from river import optim
    >>> from river import preprocessing

    >>> models = [
    ...     linear_model.LogisticRegression(optimizer=optim.SGD(lr=lr))
    ...     for lr in [0.0001, 0.001, 1e-05, 0.01]
    ... ]

    >>> dataset = datasets.Phishing()
    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     model_selection.BanditClassifier(
    ...         models,
    ...         metric=metrics.Accuracy(),
    ...         policy=bandit.Exp3(
    ...             gamma=0.5,
    ...             seed=42
    ...         )
    ...     )
    ... )
    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    Accuracy: 87.20%

    """


@pytest.mark.parametrize(
    "policy",
    [
        pytest.param(
            policy(**params),
            id=f"{policy.__name__}",
        )
        for _, policy in inspect.getmembers(
            importlib.import_module("river.bandit"),
            lambda obj: inspect.isclass(obj)
            and issubclass(obj, bandit.base.Policy)
            and not issubclass(obj, bandit.base.ContextualPolicy)
            and obj.__name__ not in {"ThompsonSampling"},
        )
        for params in policy._unit_test_params()
    ],
)
def test_bandit_classifier_with_each_policy(policy):
    models = [
        linear_model.LogisticRegression(optimizer=optim.SGD(lr=lr))
        for lr in [0.0001, 0.001, 1e-05, 0.01]
    ]

    dataset = datasets.Phishing()
    model = preprocessing.StandardScaler() | model_selection.BanditClassifier(
        models, metric=metrics.Accuracy(), policy=policy
    )
    metric = metrics.Accuracy()

    score = evaluate.progressive_val_score(dataset, model, metric)
    assert score.get() > 0.5


@pytest.mark.parametrize(
    "policy",
    [
        pytest.param(
            policy(**params),
            id=f"{policy.__name__}",
        )
        for _, policy in inspect.getmembers(
            importlib.import_module("river.bandit"),
            lambda obj: inspect.isclass(obj)
            and issubclass(obj, bandit.base.Policy)
            and not issubclass(obj, bandit.base.ContextualPolicy)
            and obj.__name__ not in {"ThompsonSampling", "Exp3"},
        )
        for params in policy._unit_test_params()
    ],
)
def test_bandit_regressor_with_each_policy(policy):
    models = [
        linear_model.LinearRegression(optimizer=optim.SGD(lr=lr))
        for lr in [0.0001, 0.001, 1e-05, 0.01]
    ]

    dataset = datasets.TrumpApproval()
    model = preprocessing.StandardScaler() | model_selection.BanditRegressor(
        models, metric=metrics.MSE(), policy=policy
    )
    metric = metrics.MSE()

    score = evaluate.progressive_val_score(dataset, model, metric)
    assert score.get() < 300
