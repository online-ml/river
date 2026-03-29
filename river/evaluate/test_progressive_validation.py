"""Regression tests for progressive_val_score optimizations."""

from __future__ import annotations

import datetime as dt

from river import datasets, evaluate, linear_model, metrics, preprocessing


def test_progressive_val_score_basic():
    """Basic progressive validation produces expected metric."""
    model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
    metric = evaluate.progressive_val_score(
        dataset=datasets.Phishing(),
        model=model,
        metric=metrics.Accuracy(),
    )
    # Sanity: accuracy should be well above random (50%)
    assert metric.get() > 0.80


def test_progressive_val_score_no_delay_matches_manual():
    """No-delay progressive validation matches manual predict-then-learn loop."""
    dataset = datasets.Phishing()

    # Method 1: progressive_val_score
    model1 = preprocessing.StandardScaler() | linear_model.LogisticRegression()
    metric1 = evaluate.progressive_val_score(
        dataset=dataset,
        model=model1,
        metric=metrics.ROCAUC(),
    )

    # Method 2: manual loop (the ground truth)
    model2 = preprocessing.StandardScaler() | linear_model.LogisticRegression()
    metric2 = metrics.ROCAUC()
    for x, y in dataset:
        y_pred = model2.predict_proba_one(x)
        metric2.update(y, y_pred)
        model2.learn_one(x, y)

    assert abs(metric1.get() - metric2.get()) < 1e-10


def test_progressive_val_score_with_delay():
    """Delayed progressive validation still works correctly with a constant delay."""
    from river import stream

    # Use simulate_qa directly to verify delayed behavior
    time_table = [
        ({"date": dt.datetime(2020, 1, 1, 20, 0, 0), "val": 1.0}, 10),
        ({"date": dt.datetime(2020, 1, 1, 20, 10, 0), "val": 2.0}, 20),
        ({"date": dt.datetime(2020, 1, 1, 20, 20, 0), "val": 3.0}, 30),
    ]

    events = list(
        stream.simulate_qa(
            time_table,
            moment="date",
            delay=dt.timedelta(minutes=5),
        )
    )
    # With 5-min delay: questions come at t=0,10,20; answers at t=5,15,25
    # So order should be: Q0, A0, Q1, A1, Q2, A2
    assert len(events) == 6
    # First event is a question (y=None)
    assert events[0][2] is None
    # Second event is the answer for sample 0
    assert events[1][0] == 0
    assert events[1][2] == 10


def test_progressive_val_score_regressor():
    """Regression task works correctly."""
    model = preprocessing.StandardScaler() | linear_model.LinearRegression()
    metric = evaluate.progressive_val_score(
        dataset=datasets.TrumpApproval(),
        model=model,
        metric=metrics.MAE(),
    )
    # MAE should be reasonable (not infinite or zero)
    assert 0.0 < metric.get() < 10.0


def test_iter_progressive_val_score():
    """iter_progressive_val_score yields correct checkpoints."""
    model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
    steps = list(
        evaluate.iter_progressive_val_score(
            dataset=datasets.Phishing(),
            model=model,
            metric=metrics.Accuracy(),
            step=500,
        )
    )
    # Should have checkpoints at 500, 1000, and final (1250)
    assert len(steps) >= 2
    assert steps[0]["Step"] == 500
    assert steps[1]["Step"] == 1000
