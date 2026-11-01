from __future__ import annotations

import math
import pickle

import pytest

from river import cluster


def test_default_configuration_uses_observation_time():
    model = cluster.TextClust()

    model.learn_one({"hello": 1.0})

    assert model.t == 1
    assert len(model.micro_clusters) == 1


def test_real_time_fading_requires_time():
    model = cluster.TextClust(real_time_fading=True)

    with pytest.raises(ValueError):
        model.learn_one({"hello": 1.0})


def test_merge_fades_combined_weight():
    model = cluster.TextClust(
        radius=1,
        fading_factor=0.1,
        tgap=100,
        real_time_fading=True,
    )

    model.learn_one({"hello": 1.0}, t=0)
    model.learn_one({"hello": 1.0}, t=10)

    micro_cluster = model.micro_clusters[0]
    assert micro_cluster.weight == pytest.approx(1.0)
    assert micro_cluster.tf["hello"] == pytest.approx(1.5)


def test_zero_fading_factor_keeps_new_clusters():
    model = cluster.TextClust(fading_factor=0, tgap=1)

    model.learn_one({"hello": 1.0})

    assert model.omega == 0
    assert len(model.micro_clusters) == 1


def test_idf_cache_is_invalidated_when_term_support_changes():
    model = cluster.TextClust(radius=0.5)
    model.learn_one({"first": 1.0})

    first_idf = model._get_idf()
    assert first_idf is model._get_idf()

    model.learn_one({"second": 1.0})

    assert model._get_idf() == {
        "first": pytest.approx(1 + math.log(2)),
        "second": pytest.approx(1 + math.log(2)),
    }


def test_term_frequencies_share_storage_with_vector_dict():
    model = cluster.TextClust(radius=1, fading_factor=0)
    model.learn_one({"first": 1.0})
    model.learn_one({"first": 1.0, "second": 2.0})

    model = pickle.loads(pickle.dumps(model))
    model.learn_one({"second": 1.0})
    micro_cluster = model.micro_clusters[0]

    assert micro_cluster._vector.to_dict() == micro_cluster.tf


def test_cleanup_uses_elapsed_time_for_real_time_fading():
    model = cluster.TextClust(real_time_fading=True, tgap=2)

    model.learn_one({"first": 1.0}, t=10_000)
    assert model.last_cleanup == 10_000

    model.learn_one({"second": 1.0}, t=10_001)
    assert model.last_cleanup == 10_000

    model.learn_one({"third": 1.0}, t=10_002)
    assert model.last_cleanup == 10_002


def test_cleanup_uses_observation_count_for_observation_time():
    model = cluster.TextClust(real_time_fading=False, tgap=2)

    model.learn_one({"first": 1.0})
    assert model.last_cleanup == 0

    model.learn_one({"second": 1.0})
    assert model.last_cleanup == 2


def test_radius_is_inclusive():
    model = cluster.TextClust(radius=1)

    model.learn_one({"first": 1.0})
    model.learn_one({"second": 1.0})

    assert len(model.micro_clusters) == 1


def test_auto_radius_uses_closest_distance_as_margin():
    class Distance:
        def __init__(self):
            self.distances = iter([0.18, 0.2, 0.4])

        def dist(self, first, second, idf):
            return next(self.distances)

    model = cluster.TextClust(auto_r=True, sigma=1)
    model.micro_clusters = {0: object(), 1: object(), 2: object()}

    cluster_id, _ = model._get_closest_mc(object(), {}, Distance())

    assert cluster_id is None


def test_auto_radius_starts_with_three_micro_clusters():
    model = cluster.TextClust(auto_r=True)

    model.learn_one({"first": 1.0})
    model.learn_one({"second": 1.0})
    model.learn_one({"third": 1.0})

    assert len(model.micro_clusters) == 3


def test_auto_merge_regularizes_single_accepted_distance():
    class Distance:
        def dist(self, first, second, idf):
            return 0.15

    model = cluster.TextClust(auto_r=True)
    model.t = 1
    model.micro_clusters = {
        index: model.MicroCluster({str(index): {"tf": 1.0}}, 1, 1, None, index)
        for index in range(2)
    }
    model._micro_distance = Distance()
    model._dist_mean = 0.2
    model._num_merged_obs = 1

    model._mergemicroclusters()

    assert len(model.micro_clusters) == 2


def test_complete_linkage_macro_clustering():
    model = cluster.TextClust()
    micro_clusters = {key: object() for key in "abcd"}
    distances = [
        [0, 1, 1.5, 5],
        [1, 0, 4, 5],
        [1.5, 4, 0, 2],
        [5, 5, 2, 0],
    ]
    model._get_distance_matrix = lambda clusters: (list(clusters), distances)

    actual = model._agglomerative_clustering(micro_clusters, 2)

    assert {frozenset(group) for group in actual} == {
        frozenset({"a", "b"}),
        frozenset({"c", "d"}),
    }


def test_single_micro_cluster_builds_macro_cluster():
    model = cluster.TextClust()
    model.learn_one({"hello": 1.0})

    macro_clusters = model.get_macroclusters()

    assert len(macro_clusters) == 1
    assert macro_clusters[0].merged_ids == [0]


def test_empty_model_has_no_macro_clusters():
    model = cluster.TextClust()

    assert model.get_macroclusters() == {}


def test_macro_prediction_accepts_micro_cluster_zero():
    model = cluster.TextClust(radius=0.1, num_macro=1)
    model.learn_one({"first": 1.0})
    model.learn_one({"second": 1.0})

    assert model.predict_one({"first": 1.0}, type="macro") == 0


def test_macro_clusters_only_include_eligible_micro_clusters():
    model = cluster.TextClust(radius=0.1, min_weight=1.5, num_macro=3)
    model.learn_one({"frequent": 1.0})
    model.learn_one({"frequent": 1.0})
    model.learn_one({"rare": 1.0})

    macro_clusters = model.get_macroclusters()

    assert len(macro_clusters) == 1
    assert macro_clusters[0].merged_ids == [0]


def test_distance_is_finite_for_identical_documents():
    model = cluster.TextClust()
    first = model.MicroCluster({"hello": {"tf": 1.0}}, 1, 1, None, 0)
    second = model.MicroCluster({"hello": {"tf": 1.0}}, 1, 1, None, 1)

    distance = model._micro_distance.dist(first, second, {"hello": 1.0})

    assert math.isfinite(distance)
    assert distance == 0
