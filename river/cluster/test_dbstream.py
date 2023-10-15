from __future__ import annotations

import pytest

from river.cluster import DBSTREAM


def build_dbstream(fading_factor=0.001, intersection_factor=0.05):
    return DBSTREAM(
        fading_factor=fading_factor,
        clustering_threshold=1,
        cleanup_interval=1,
        intersection_factor=intersection_factor,
    )


def add_cluster(dbstream, initial_point, move_towards, times=1):
    dbstream.learn_one(initial_point)
    for _ in range(times):
        dbstream.learn_one(move_towards)


def assert_micro_cluster_properties(cluster, center, last_update=None):
    assert cluster.center == pytest.approx(center)
    if last_update is not None:
        assert cluster.last_update == last_update


def test_cluster_formation_and_cleanup():
    dbstream = build_dbstream()

    X = [
        {1: 1},
        {1: 3},
        {1: 3},
        {1: 3},
        {1: 5},
        {1: 7},
        {1: 9},
        {1: 11},
        {1: 11},
        {1: 13},
        {1: 11},
        {1: 15},
        {1: 17},
    ]

    for x in X:
        dbstream.learn_one(x)

    assert len(dbstream._micro_clusters) == 3
    assert_micro_cluster_properties(dbstream.micro_clusters[1], center={1: 3}, last_update=3)
    assert_micro_cluster_properties(dbstream.micro_clusters[5], center={1: 11}, last_update=10)
    assert_micro_cluster_properties(dbstream.micro_clusters[7], center={1: 17}, last_update=12)


def test_with_two_micro_clusters():
    dbstream = build_dbstream()

    add_cluster(dbstream, initial_point={1: 1, 2: 1}, move_towards={1: 1.7, 2: 1.7}, times=25)
    add_cluster(dbstream, initial_point={1: 3, 2: 3}, move_towards={1: 2.3, 2: 2.3}, times=25)
    # Points in the middle of first and second micro-clusters
    for _ in range(5):
        dbstream.learn_one({1: 2, 2: 2})

    assert len(dbstream._micro_clusters) == 2
    assert_micro_cluster_properties(
        dbstream.micro_clusters[0], center={1: 1.597322, 2: 1.597322}, last_update=56
    )
    assert_micro_cluster_properties(
        dbstream.micro_clusters[1], center={1: 2.402677, 2: 2.402677}, last_update=56
    )

    assert dbstream.s == {0: {1: 3.995844478090532}}
    assert dbstream.s_t == {0: {1: 56}}

    dbstream._recluster()
    assert len(dbstream.clusters) == 1
    assert_micro_cluster_properties(dbstream.clusters[0], center={1: 2.003033, 2: 2.003033})


def test_density_graph_with_three_micro_clusters():
    dbstream = build_dbstream()

    add_cluster(dbstream, initial_point={1: 1, 2: 1}, move_towards={1: 1.7, 2: 1.7}, times=25)
    add_cluster(dbstream, initial_point={1: 3, 2: 3}, move_towards={1: 2.3, 2: 2.3}, times=25)
    # Points in the middle of first and second micro-clusters
    for _ in range(5):
        dbstream.learn_one({1: 2, 2: 2})

    add_cluster(dbstream, initial_point={1: 4, 2: 4}, move_towards={1: 3.3, 2: 3.3}, times=25)
    # Points in the middle of second and third micro-clusters
    for _ in range(4):
        dbstream.learn_one({1: 3, 2: 3})

    assert len(dbstream._micro_clusters) == 3

    assert_micro_cluster_properties(
        dbstream.micro_clusters[0], center={1: 1.597322, 2: 1.597322}, last_update=56
    )
    assert_micro_cluster_properties(
        dbstream.micro_clusters[1], center={1: 2.461654, 2: 2.461654}, last_update=86
    )
    assert_micro_cluster_properties(
        dbstream.micro_clusters[2], center={1: 3.430485, 2: 3.430485}, last_update=86
    )

    assert dbstream.s[0] == pytest.approx({1: 3.995844})
    assert dbstream.s[1] == pytest.approx({2: 2.997921})
    assert dbstream.s_t == {0: {1: 56}, 1: {2: 86}}

    dbstream._recluster()
    assert len(dbstream.clusters) == 1
    assert_micro_cluster_properties(dbstream.clusters[0], center={1: 2.489894, 2: 2.489894})


def test_density_graph_with_removed_microcluster():
    dbstream = build_dbstream(fading_factor=0.1, intersection_factor=0.3)

    add_cluster(dbstream, initial_point={1: 1, 2: 1}, move_towards={1: 1.7, 2: 1.7}, times=25)
    add_cluster(dbstream, initial_point={1: 3, 2: 3}, move_towards={1: 2.3, 2: 2.3}, times=25)
    # Points in the middle of first and second micro-clusters
    for _ in range(5):
        dbstream.learn_one({1: 2, 2: 2})

    add_cluster(dbstream, initial_point={1: 4, 2: 4}, move_towards={1: 3.3, 2: 3.3}, times=25)
    # Points in the middle of second and third micro-clusters
    for _ in range(4):
        dbstream.learn_one({1: 3, 2: 3})

    assert len(dbstream._micro_clusters) == 2
    assert_micro_cluster_properties(
        dbstream.micro_clusters[1], center={1: 2.461654, 2: 2.461654}, last_update=86
    )
    assert_micro_cluster_properties(
        dbstream.micro_clusters[2], center={1: 3.430485, 2: 3.430485}, last_update=86
    )

    assert dbstream.s[0] == pytest.approx({1: 3.615835})
    assert dbstream.s[1] == pytest.approx({2: 2.803583})
    assert dbstream.s_t == {0: {1: 56}, 1: {2: 86}}

    dbstream._recluster()
    assert len(dbstream.clusters) == 1
    assert_micro_cluster_properties(dbstream.clusters[0], center={1: 3.152231, 2: 3.152231})
