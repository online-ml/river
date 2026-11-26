from __future__ import annotations

import copy
import pickle
import random

import numpy as np
import pandas as pd

from river import checks, cluster, covariance, datasets


def _assert_undirected_invariant(gng: cluster.GNG) -> None:
    unit_set = set(gng._units)
    for u in gng._units:
        for e in u.neighbors:
            other = e.p1 if e.p0 is u else e.p0
            assert other in unit_set, "Edge points to a removed unit"
            assert e in other.neighbors, "Directed/asymmetric edge found"


def test_gng_river_estimator_checks():
    model = cluster.GNG()
    for check in (
        checks.common.check_repr,
        checks.common.check_str,
        checks.common.check_tags,
        checks.common.check_clone_same_class,
        checks.common.check_clone_is_idempotent,
        checks.common.check_init_has_default_params_for_tests,
        checks.common.check_init_default_params_are_not_mutable,
        checks.common.check_doc,
        checks.common.check_clone_changes_memory_addresses,
        checks.common.check_mutate_can_be_idempotent,
    ):
        check(model.clone())

    try:
        import sklearn  # noqa: F401
    except ImportError:
        pass
    else:
        checks.check_estimator(model)


def test_gng_river_stream_invariants():
    data = list(datasets.Phishing().take(100))
    model = cluster.GNG(lambda_=10, max_age=50)

    # 1. check_learn_one (returns self and does not mutate input)
    checks.common.check_learn_one(model.clone(), data)

    # 2. Shuffled feature keys produce identical predictions and prototypes
    rng = random.Random(42)
    m_orig = model.clone()
    m_shuf = model.clone()
    for x, _ in data:
        keys = list(x.keys())
        rng.shuffle(keys)
        x_shuf = {k: x[k] for k in keys}
        assert m_orig.predict_one(x) == m_shuf.predict_one(x_shuf)
        m_orig.learn_one(x)
        m_shuf.learn_one(x_shuf)
    assert np.array_equal(m_orig.prototypes_array, m_shuf.prototypes_array)

    # 3. Emerging and disappearing features
    m_dyn = model.clone()
    for i, (x, _) in enumerate(data):
        keys = list(x.keys())
        sub = {k: x[k] for k in keys[:-3]} if i % 2 == 0 else x
        assert isinstance(m_dyn.predict_one(x), int)
        m_dyn.learn_one(sub)
        _assert_undirected_invariant(m_dyn)

    # 4. Pickling before and after training
    m = model.clone()
    assert isinstance(pickle.loads(pickle.dumps(m)), cluster.GNG)
    for x, _ in data:
        m.learn_one(x)
    restored = pickle.loads(pickle.dumps(m))
    assert restored.n_clusters == m.n_clusters
    assert np.array_equal(restored.prototypes_array, m.prototypes_array)


def test_gng_undirected_invariant_and_max_nodes():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(300, 4))
    gng = cluster.GNG(lambda_=5, max_age=25, max_nodes=20)
    for row in X:
        gng.learn_array(row)
        _assert_undirected_invariant(gng)
        assert gng.n_clusters <= 20
    assert gng.n_clusters > 2
    assert gng.n_samples_seen == 300


def test_gng_clone_resets_and_deepcopy_preserves_state():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(150, 5))
    cols = [f"f{i}" for i in range(5)]
    emp_cov = covariance.EmpiricalCovariance(ddof=0)
    emp_cov.update_many(pd.DataFrame(X, columns=cols))
    inv_cov = emp_cov.precision(cols)

    gng = cluster.GNG(lambda_=5, max_age=50, dist_metric="mahalanobis", inv_cov=inv_cov)
    for row in X[:100]:
        gng.learn_array(row)

    # River clone() must return a fresh untrained estimator
    fresh = gng.clone()
    assert fresh.n_clusters == 0
    assert fresh.n_samples_seen == 0

    # copy.deepcopy() must preserve exact graph state and future trajectory bit-for-bit
    cloned = copy.deepcopy(gng)
    assert np.array_equal(gng.prototypes_array, cloned.prototypes_array)
    for row in X[100:]:
        gng.learn_array(row)
        cloned.learn_array(row)
    assert np.array_equal(gng.prototypes_array, cloned.prototypes_array)
    assert gng.n_prototypes_created == cloned.n_prototypes_created
    _assert_undirected_invariant(cloned)
