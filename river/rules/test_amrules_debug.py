"""Temporary debug to isolate why amrules.anomaly_score doctest diverges on Linux CI under pytest 9."""

from __future__ import annotations

import math
import sys


def test_amrules_doctest_debug():
    from river import drift, rules, tree
    from river.datasets import synth

    print("\n=== AMRules CI debug ===", flush=True)
    print(f"python: {sys.version}", flush=True)
    print(f"platform: {sys.platform}", flush=True)
    print(f"float_repr: {sys.float_repr_style}", flush=True)
    print(f"hash_seed: {sys.flags.hash_randomization}, hash_info: {sys.hash_info}", flush=True)
    print(f"flags: {sys.flags}", flush=True)
    print(f"math.pi repr: {math.pi!r}", flush=True)

    # Probe libm determinism at the values Friedman will produce
    import random

    rng_probe = random.Random(42)
    sample_xs = [rng_probe.uniform(0, 1) for _ in range(10)]
    sample_ys = [rng_probe.gauss(0, 1) for _ in range(5)]
    print(f"probe uniform: {sample_xs!r}", flush=True)
    print(f"probe gauss: {sample_ys!r}", flush=True)
    print(f"sin(pi*0.5*0.5): {math.sin(math.pi * 0.5 * 0.5)!r}", flush=True)

    dataset = synth.Friedman(seed=42).take(1001)
    model = rules.AMRules(
        n_min=50,
        delta=0.1,
        drift_detector=drift.ADWIN(),
        splitter=tree.splitter.QOSplitter(),
    )

    last_x = None
    for i, (x, y) in enumerate(dataset):
        if i == 0:
            print(f"first sample x: {x!r}", flush=True)
            print(f"first sample y: {y!r}", flush=True)
        if i in (10, 100, 500, 999):
            print(
                f"i={i} n_rules={len(model._rules)} default_rule_weight={model._default_rule.total_weight}",
                flush=True,
            )
        if i == 1000:
            last_x = x
            break
        model.learn_one(x, y)

    print(f"final n_rules={len(model._rules)}", flush=True)
    print(f"last sample x (not trained): {last_x!r}", flush=True)
    for rule_id, rule in model._rules.items():
        print(f"rule id={rule_id} literals={[lit.describe() for lit in rule.literals]}", flush=True)
        print(f"  covers={rule.covers(last_x)} score_one={rule.score_one(last_x)!r}", flush=True)
        print(f"  total_weight={rule.total_weight}", flush=True)
        for feat, fs in rule._feat_stats.items():
            print(f"  feat {feat}: mean={fs.mean.get()!r} var={fs.get()!r}", flush=True)

    score = model.anomaly_score(last_x)
    print(f"anomaly_score: {score!r}", flush=True)
    print(f"n_drifts_detected: {model.n_drifts_detected}", flush=True)
    print("=== end debug ===", flush=True)

    # Force fail so pytest dumps the captured stdout
    assert score == (0, 0, 0), "intentional fail to expose captured stdout"
