from __future__ import annotations

import typing

import pytest

from river import drift, rules, tree
from river.datasets import synth
from river.rules import amrules

if typing.TYPE_CHECKING:
    from collections.abc import Iterator

    from river.datasets.base import Dataset

Sample: typing.TypeAlias = tuple[dict[int, float], float]

DRIFTING_STREAMS: dict[str, tuple[Dataset, Dataset, float]] = {
    "planes2d-then-friedman": (synth.Planes2D(seed=42), synth.Friedman(seed=42), 1.0),
    "friedman-then-scaled-friedman": (synth.Friedman(seed=42), synth.Friedman(seed=43), 10.0),
    "planes2d-then-scaled-planes2d": (synth.Planes2D(seed=42), synth.Planes2D(seed=43), 10.0),
}


def make_model(ordered_rule_set: bool = True) -> rules.AMRules:
    return rules.AMRules(
        n_min=50,
        delta=0.1,
        drift_detector=drift.ADWIN(),
        splitter=tree.splitter.QOSplitter(),
        ordered_rule_set=ordered_rule_set,
    )


def fit_and_score(model: rules.AMRules) -> tuple[float, float, float]:
    samples: list[Sample] = list(synth.Friedman(seed=42).take(1001))
    for x, y in samples[:-1]:
        model.learn_one(x, y)
    return model.anomaly_score(samples[-1][0])


def abrupt_drift(before: Dataset, after: Dataset, scale: float) -> Iterator[Sample]:
    yield from before.take(2000)
    drifted: list[Sample] = list(after.take(2000))
    for x, y in drifted:
        yield x, scale * y


def same_hash(rule: amrules.RegRule) -> int:
    return 0


@pytest.mark.parametrize("ordered_rule_set", [True, False], ids=["ordered", "unordered"])
@pytest.mark.parametrize(
    ("before", "after", "scale"), DRIFTING_STREAMS.values(), ids=DRIFTING_STREAMS
)
def test_rules_are_only_dropped_by_drift(
    before: Dataset,
    after: Dataset,
    scale: float,
    ordered_rule_set: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(amrules.RegRule, "__hash__", same_hash)
    model = make_model(ordered_rule_set)
    default_rule = model._default_rule
    n_created = 0

    for x, y in abrupt_drift(before, after, scale):
        model.learn_one(x, y)
        if model._default_rule is not default_rule:
            default_rule = model._default_rule
            n_created += 1

    assert model.n_drifts_detected > 0
    assert len(model._rules) == n_created - model.n_drifts_detected


def test_new_rule_does_not_overwrite_a_live_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    # See https://github.com/online-ml/river/issues/2052
    expected = fit_and_score(make_model())
    monkeypatch.setattr(amrules.RegRule, "__hash__", same_hash)

    assert fit_and_score(make_model()) == expected
