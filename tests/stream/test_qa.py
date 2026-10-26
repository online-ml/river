from __future__ import annotations

import datetime as dt

import pytest

from river import active, compose, evaluate, linear_model, metrics, stream


def make_dataset():
    return [({"t": dt.datetime(2020, 1, day), "a": day}, day % 2) for day in range(1, 6)]


def expected_events(dataset):
    events = []
    for i, (x, y) in enumerate(dataset):
        events.append((i, x, None))
        events.append((i, x, y))
    return events


def make_model():
    return compose.Select("a") | linear_model.LogisticRegression()


def moment_from_features(x):
    return x["t"]


@pytest.mark.parametrize("moment", [None, "t", moment_from_features])
def test_simulate_qa_delay_none_with_moment(moment):
    dataset = make_dataset()
    events = list(stream.simulate_qa(dataset, moment=moment, delay=None))
    assert events == expected_events(dataset)


def test_simulate_qa_delay_none_with_kwargs():
    dataset = [({"t": dt.datetime(2020, 1, 1)}, 1, {"w": 2})]
    events = list(stream.simulate_qa(dataset, moment="t", delay=None))
    assert events == [
        (0, {"t": dt.datetime(2020, 1, 1)}, None, {"w": 2}),
        (0, {"t": dt.datetime(2020, 1, 1)}, 1, {"w": 2}),
    ]


def test_simulate_qa_delay_none_copies_questions():
    dataset = make_dataset()
    events = list(stream.simulate_qa(dataset, moment="t", delay=None, copy=True))
    assert events[0][1] is not events[1][1]
    events = list(stream.simulate_qa(dataset, moment="t", delay=None, copy=False))
    assert events[0][1] is events[1][1]


def make_active_model():
    return active.EntropySampler(make_model(), seed=42)


@pytest.mark.parametrize("model_factory", [make_model, make_active_model], ids=["plain", "active"])
@pytest.mark.parametrize("moment", ["t", moment_from_features])
def test_progressive_val_score_delay_none_with_moment(model_factory, moment):
    dataset = make_dataset()
    with_moment = evaluate.progressive_val_score(
        dataset, model_factory(), metrics.Accuracy(), moment=moment
    )
    without_moment = evaluate.progressive_val_score(dataset, model_factory(), metrics.Accuracy())
    assert with_moment.get() == without_moment.get()
