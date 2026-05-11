"""Temporary debug: same code in 3 different contexts to isolate where divergence happens.

Hypothesis: the AMRules.anomaly_score doctest fails on Linux CI because of
something pytest's doctest plugin does, not because of platform/order. The
debug function passed in the same CI process, so re-running the exact same
training code from inside a doctest in *this* file should reveal whether the
context (doctest vs function) is what flips the result.

>>> from river import drift, rules, tree
>>> from river.datasets import synth

>>> dataset = synth.Friedman(seed=42).take(1001)

>>> model = rules.AMRules(
...     n_min=50,
...     delta=0.1,
...     drift_detector=drift.ADWIN(),
...     splitter=tree.splitter.QOSplitter()
... )

>>> for i, (x, y) in enumerate(dataset):
...     if i == 1000:
...         break
...     model.learn_one(x, y)

>>> model.anomaly_score(x)
(1.0168907243483933, 0.13045786430817402, 1.0)

"""

from __future__ import annotations

import math
import sys


def test_amrules_function():
    """Same exact code as the failing doctest, but as a regular function."""
    from river import drift, rules, tree
    from river.datasets import synth

    print(f"\n[function] python: {sys.version}", flush=True)
    print(f"[function] platform: {sys.platform}", flush=True)

    dataset = synth.Friedman(seed=42).take(1001)
    model = rules.AMRules(
        n_min=50,
        delta=0.1,
        drift_detector=drift.ADWIN(),
        splitter=tree.splitter.QOSplitter(),
    )

    last_x = None
    for i, (x, y) in enumerate(dataset):
        if i == 1000:
            last_x = x
            break
        model.learn_one(x, y)

    score = model.anomaly_score(last_x)
    print(f"[function] anomaly_score: {score!r}", flush=True)

    assert score == (0, 0, 0), f"intentional fail to dump output, got {score!r}"


def test_with_explicit_math_check():
    """Probe whether math.log returns the same value as in the doctest's score path."""
    from river import drift, rules, tree
    from river.datasets import synth

    # Probe the inner expression used by RegRule.score_one: log(p) - log(1-p)
    p = 0.6
    val = math.log(p) - math.log(1 - p)
    print(f"\n[probe] math.log(0.6) - math.log(0.4) = {val!r}", flush=True)
    print(f"[probe] math.log(0.6) = {math.log(0.6)!r}", flush=True)
    print(f"[probe] math.log(0.4) = {math.log(0.4)!r}", flush=True)

    dataset = synth.Friedman(seed=42).take(1001)
    model = rules.AMRules(
        n_min=50,
        delta=0.1,
        drift_detector=drift.ADWIN(),
        splitter=tree.splitter.QOSplitter(),
    )

    last_x = None
    for i, (x, y) in enumerate(dataset):
        if i == 1000:
            last_x = x
            break
        model.learn_one(x, y)

    score = model.anomaly_score(last_x)
    assert score == (0, 0, 0), f"intentional fail to dump output, got {score!r}"
