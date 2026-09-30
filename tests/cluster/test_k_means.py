from __future__ import annotations

import collections

import pytest

from river import cluster


def reference_predict(model, x):
    def distance(c):
        center = model.centers[c]
        return sum((abs(center[k] - x.get(k, 0))) ** model.p for k in {*center.keys(), *x.keys()})

    return min(model.centers, key=distance)


@pytest.mark.parametrize("p", [1, 2, 3])
@pytest.mark.parametrize("sparse", [False, True])
def test_predictions_match_reference(p, sparse):
    actual = cluster.KMeans(n_clusters=5, p=p, seed=42)
    reference = cluster.KMeans(n_clusters=5, p=p, seed=42)
    samples = [
        {i: ((t * 17 + i * 11) % 31) / 7 for i in range(32) if not sparse or (i + t) % 5 == 0}
        for t in range(50)
    ]

    for x in samples:
        expected = reference_predict(reference, x)
        assert actual.learn_predict_one(x) == expected
        for i, xi in x.items():
            reference.centers[expected][i] += reference.halflife * (
                xi - reference.centers[expected][i]
            )
        assert actual.centers == reference.centers
        assert actual._rng.getstate() == reference._rng.getstate()


def test_non_dict_mapping_matches_reference():
    actual = cluster.KMeans(n_clusters=3, seed=17)
    reference = cluster.KMeans(n_clusters=3, seed=17)
    x = collections.UserDict({"a": 2.0, "b": -1.0})

    assert actual.predict_one(x) == reference_predict(reference, x)
    assert actual.centers == reference.centers
    assert actual._rng.getstate() == reference._rng.getstate()


def test_empty_sample_matches_reference():
    actual = cluster.KMeans(n_clusters=3, seed=17)
    reference = cluster.KMeans(n_clusters=3, seed=17)

    assert actual.predict_one({}) == reference_predict(reference, {})
    assert actual.centers == reference.centers
