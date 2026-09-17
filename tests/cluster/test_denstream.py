from __future__ import annotations

import math

import pytest

from river import cluster
from river.cluster.denstream import DenStreamMicroCluster


def build_micro_cluster(points):
    mc = DenStreamMicroCluster(x=points[0], timestamp=0, decaying_factor=0.01)
    for x in points[1:]:
        mc.insert(x, 0)
    return mc


def square(cx, cy, half_side=1):
    # The four corners are all at the same distance from the center, so the
    # radius of a micro-cluster built from them is exactly that distance.
    return [
        {0: cx + dx, 1: cy + dy} for dx in (-half_side, half_side) for dy in (-half_side, half_side)
    ]


def test_radius():
    mc = build_micro_cluster(square(0, 0))
    assert mc.center == pytest.approx({0: 0, 1: 0})
    assert mc.calc_radius(0) == pytest.approx(math.sqrt(2))


def test_radius_does_not_depend_on_position():
    """See https://github.com/online-ml/river/issues/2004"""
    for cx, cy in [(1000, 0), (0, 1000), (1000, 1000), (-23000, 1700)]:
        mc = build_micro_cluster(square(cx, cy))
        assert mc.center == pytest.approx({0: cx, 1: cy})
        assert mc.calc_radius(0) == pytest.approx(math.sqrt(2))


def test_radius_with():
    mc = build_micro_cluster(square(1000, 1000))
    x = {0: 1003, 1: 1000}
    radius = mc.radius_with(x)
    mc.insert(x, 0)
    assert mc.calc_radius(0) == pytest.approx(radius)


def test_far_point_makes_a_new_micro_cluster():
    denstream = cluster.DenStream(
        decaying_factor=0.01, beta=1, mu=1.0005, epsilon=1, n_samples_init=4
    )

    for x in square(-23000, 1700, half_side=0.25):
        denstream.learn_one(x)
    assert len(denstream.p_micro_clusters) == 1

    # 50 away from the only micro-cluster: it can't fit within epsilon
    for x in square(-22950, 1700, half_side=0.25):
        denstream.learn_one(x)
    assert len(denstream.p_micro_clusters) == 2


def test_clusters_expand_past_the_first_hop():
    """See https://github.com/online-ml/river/issues/2006"""
    denstream = cluster.DenStream(decaying_factor=0.01, beta=1, mu=1.0005, epsilon=0.4)
    denstream.initialized = True

    # Three micro-clusters on a line: each one reaches the next, the first
    # doesn't reach the last.
    a, b, c = (build_micro_cluster(square(cx, 0, half_side=0.25)) for cx in (0, 0.5, 1))
    denstream.p_micro_clusters = {0: a, 1: b, 2: c}
    assert denstream._is_directly_density_reachable(a, b)
    assert denstream._is_directly_density_reachable(b, c)
    assert not denstream._is_directly_density_reachable(a, c)

    denstream.predict_one({0: 0, 1: 0})
    assert denstream.n_clusters == 1


def test_promoted_micro_cluster_does_not_overwrite_another():
    """See point 2 of https://github.com/online-ml/river/discussions/1555"""
    denstream = cluster.DenStream(decaying_factor=0.01, beta=1, mu=1.0005, epsilon=1)
    denstream.initialized = True
    denstream.p_micro_clusters = {
        i: build_micro_cluster(square(cx, 0, half_side=0.25)) for i, cx in enumerate((0, 10, 20))
    }
    # The one in the middle gets pruned, the last one keeps key 2.
    del denstream.p_micro_clusters[1]

    # Far from everything: the first point starts an outlier micro-cluster,
    # the second one promotes it.
    denstream.learn_one({0: 100, 1: 0})
    denstream.learn_one({0: 100.1, 1: 0})
    assert len(denstream.p_micro_clusters) == 3
