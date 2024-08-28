from __future__ import annotations

import pytest
from sklearn.datasets import make_blobs

from river import metrics, stream, utils
from river.cluster import DBSTREAM


def build_dbstream(fading_factor=0.01, intersection_factor=0.05):
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
        {1: 2},
        {1: 3},
        {1: 3},
        {1: 3},
        {1: 5},
        {1: 7},
        {1: 9},
        {1: 10},
        {1: 11},
        {1: 11},
        {1: 12},
        {1: 13},
        {1: 11},
        {1: 15},
        {1: 15},
        {1: 16},
        {1: 17},
        {1: 17},
        {1: 17},
    ]

    for x in X:
        dbstream.learn_one(x)

    assert len(dbstream._micro_clusters) == 4
    assert_micro_cluster_properties(dbstream.micro_clusters[2], center={1: 3}, last_update=4)
    assert_micro_cluster_properties(dbstream.micro_clusters[7], center={1: 11}, last_update=13)
    assert_micro_cluster_properties(dbstream.micro_clusters[8], center={1: 15}, last_update=15)
    assert_micro_cluster_properties(dbstream.micro_clusters[10], center={1: 17}, last_update=19)

    assert dbstream.predict_one({1: 2.0}) == 0
    assert dbstream.predict_one({1: 13.0}) == 1
    assert dbstream.predict_one({1: 13 + 1e-10}) == 2
    assert dbstream.predict_one({1: 16 - 1e-10}) == 2
    assert dbstream.predict_one({1: 18}) == 3

    assert len(dbstream._clusters) == 4
    assert dbstream.s == dbstream.s_t == {}


def test_with_two_micro_clusters():
    dbstream = build_dbstream()

    add_cluster(dbstream, initial_point={1: 1, 2: 1}, move_towards={1: 1.7, 2: 1.7}, times=25)
    add_cluster(dbstream, initial_point={1: 3, 2: 3}, move_towards={1: 2.3, 2: 2.3}, times=25)

    assert len(dbstream.micro_clusters) == 2
    assert_micro_cluster_properties(
        dbstream.micro_clusters[0], center={1: 2.137623, 2: 2.137623}, last_update=51
    )
    assert_micro_cluster_properties(
        dbstream.micro_clusters[1], center={1: 2.914910, 2: 2.914910}, last_update=51
    )

    assert dbstream.s == {0: {1: 23.033438964246173}}
    assert dbstream.s_t == {0: {1: 51}}

    dbstream._recluster()
    assert len(dbstream.clusters) == 1
    assert_micro_cluster_properties(dbstream.clusters[0], center={1: 2.415239, 2: 2.415239})


def test_density_graph_with_three_micro_clusters():
    dbstream = build_dbstream()

    add_cluster(dbstream, initial_point={1: 1, 2: 1}, move_towards={1: 1.7, 2: 1.7}, times=25)
    add_cluster(dbstream, initial_point={1: 3, 2: 3}, move_towards={1: 2.3, 2: 2.3}, times=25)
    # Points in the middle of first and second micro-clusters
    for _ in range(5):
        dbstream.learn_one({1: 2, 2: 2})

    assert dbstream.s == {0: {1: 23.033438964246173}}
    assert dbstream.s_t == {0: {1: 51}}

    add_cluster(dbstream, initial_point={1: 4, 2: 4}, move_towards={1: 3.3, 2: 3.3}, times=25)
    # Points in the middle of second and third micro-clusters
    for _ in range(4):
        dbstream.learn_one({1: 3, 2: 3})

    assert len(dbstream._micro_clusters) == 3
    assert_micro_cluster_properties(
        dbstream.micro_clusters[0], center={1: 2.0, 2: 2.0}, last_update=56
    )
    assert_micro_cluster_properties(
        dbstream.micro_clusters[1], center={1: 3.0, 2: 3.0}, last_update=86
    )
    assert_micro_cluster_properties(
        dbstream.micro_clusters[2], center={1: 3.982141, 2: 3.982141}, last_update=82
    )

    assert dbstream.s[0] == pytest.approx({1: 23.033439})
    assert dbstream.s[1] == pytest.approx({2: 23.033439})
    assert dbstream.s_t == {0: {1: 51}, 1: {2: 82}}

    dbstream._recluster()
    assert len(dbstream.clusters) == 1
    print(dbstream.clusters[0].center)
    assert_micro_cluster_properties(dbstream.clusters[0], center={1: 2.800788, 2: 2.800788})


def test_density_graph_with_removed_microcluster():
    dbstream = build_dbstream(fading_factor=0.1, intersection_factor=0.3)

    add_cluster(dbstream, initial_point={1: 1, 2: 1}, move_towards={1: 1.7, 2: 1.7}, times=25)
    add_cluster(dbstream, initial_point={1: 3, 2: 3}, move_towards={1: 2.3, 2: 2.3}, times=25)
    # Points in the middle of first and second micro-clusters
    for _ in range(5):
        dbstream.learn_one({1: 2, 2: 2})

    add_cluster(dbstream, initial_point={1: 3.5, 2: 3.5}, move_towards={1: 2.9, 2: 2.9}, times=25)

    # Points in the middle of second and third micro-clusters
    for _ in range(4):
        dbstream.learn_one({1: 2.6, 2: 2.6})

    assert len(dbstream._micro_clusters) == 2
    assert_micro_cluster_properties(
        dbstream.micro_clusters[0], center={1: 2.023498, 2: 2.023498}, last_update=86
    )
    assert_micro_cluster_properties(
        dbstream.micro_clusters[1], center={1: 2.766543, 2: 2.766543}, last_update=86
    )

    assert dbstream.s == {0: {1: 4.702391097045977}}
    assert dbstream.s_t == {0: {1: 86}}

    dbstream._recluster()
    assert len(dbstream.clusters) == 1
    assert_micro_cluster_properties(dbstream.clusters[0], center={1: 2.560647, 2: 2.560647})


def test_dbstream_synthetic_sklearn():
    centers = [(-10, -10), (-5, -5), (0, 0), (5, 5), (10, 10)]
    cluster_std = [0.6] * 5

    # Create a dataset with 15000 data points with 5 centers and cluster SD of 0.6 each
    X, y = make_blobs(
        n_samples=15_000, cluster_std=cluster_std, centers=centers, n_features=2, random_state=42
    )

    dbstream = DBSTREAM(
        clustering_threshold=2,
        fading_factor=0.05,
        intersection_factor=0.1,
        cleanup_interval=1.0,
        minimum_weight=1.0,
    )

    # Use VBeta as the metric to investigate the performance of DBSTREAM
    v_beta = metrics.VBeta(beta=1.0)

    for x, y_true in stream.iter_array(X, y):
        dbstream.learn_one(x)
        y_pred = dbstream.predict_one(x)
        v_beta.update(y_true, y_pred)

    assert len(dbstream._micro_clusters) == 12
    assert round(v_beta.get(), 4) == 0.9816

    assert dbstream.s.keys() == dbstream.s_t.keys()

    dbstream._recluster()

    # Check that the resulted cluster centers are close to the expected centers
    dbstream_expected_centers = {
        0: {0: 10, 1: 10},
        1: {0: -5, 1: -5},
        2: {0: 0, 1: 0},
        3: {0: 5, 1: 5},
        4: {0: -10, 1: -10},
    }

    for i in dbstream.centers.keys():
        assert (
            utils.math.minkowski_distance(dbstream.centers[i], dbstream_expected_centers[i], 2)
            < 0.2
        )
