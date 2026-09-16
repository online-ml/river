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
