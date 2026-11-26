from __future__ import annotations

import numpy as np

from river import checks, datasets, drift, ensemble, tree


def test_groch_river_estimator_checks():
    model = ensemble.GroCHClassifier(log_level=0)

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

    binary_data = list(datasets.Phishing().take(150))
    binary_np_bool = [(x, np.bool_(y)) for x, y in binary_data]
    multiclass_data = list(datasets.ImageSegments().take(150))

    for data in (binary_data, binary_np_bool, multiclass_data):
        checks.common.check_learn_one(model.clone(), data)
        checks.common.check_pickling(model.clone(), data)
        checks.common.check_emerging_features(model.clone(), data)
        checks.common.check_disappearing_features(model.clone(), data)
        checks.clf.check_predict_proba_one(model.clone(), data)

    # Shuffle-feature invariance (use leaf_prediction='mc' on ImageSegments because
    # HoeffdingTreeClassifier's Gaussian NB underflows near 1e-298 on 19 features)
    checks.common.check_shuffle_features_no_impact(model.clone(), binary_data)
    checks.common.check_shuffle_features_no_impact(model.clone(), binary_np_bool)
    mc_groch = ensemble.GroCHClassifier(
        model=tree.HoeffdingTreeClassifier(leaf_prediction="mc"), log_level=0
    )
    checks.common.check_shuffle_features_no_impact(mc_groch, multiclass_data)

    try:
        import sklearn  # noqa: F401
    except ImportError:
        pass
    else:
        checks.check_estimator(model)


def test_groch_pre_populate_and_replacement_policies():
    data = list(datasets.Phishing().take(80))
    concept_a = data[:40]
    concept_b = data[40:]
    base_model = tree.HoeffdingTreeClassifier(leaf_prediction="nba")

    for policy in ("fifo", "least used model"):
        groch = ensemble.GroCHClassifier(
            model=base_model,
            omega=2,
            sigma=2.0,
            drift_detector=drift.binary.RDDM(n_min=20),
            pre_training_data=[(concept_a, base_model), (concept_b, base_model)],
            group_replacement_policy=policy,
            log_level=0,
        )
        assert len(groch.CH) == 2
        for x, y in data:
            groch.learn_one(x, y)
            assert groch.predict_one(x) is not None
        # Trigger eviction directly on group 0 to verify policy behavior
        g0 = groch.CH[0]
        g0["classifiers"].append(groch.c.clone())
        g0["classifiers"].append(groch.c.clone())
        groch._remove_classifier(g0)
        assert len(g0["classifiers"]) == 2


def test_groch_native_vectorized_mahalanobis_distillator():
    import pandas as pd

    from river import cluster, covariance

    rng = np.random.default_rng(7)
    X = rng.normal(size=(120, 4))
    y_vals = (X[:, 0] + X[:, 1] > 0).astype(int)
    features = ["f0", "f1", "f2", "f3"]
    cols = features + ["__label__"]
    data_mat = np.column_stack([X, y_vals.astype(np.float64)])
    emp_cov = covariance.EmpiricalCovariance(ddof=0)
    emp_cov.update_many(pd.DataFrame(data_mat, columns=cols))
    inv_cov = emp_cov.precision(cols)

    gng = cluster.GNG(lambda_=5, max_age=50, dist_metric="mahalanobis", inv_cov=inv_cov)
    dist = ensemble.TopologyDistillator(
        clusterer=gng,
        feature_subset=features,
        include_target=True,
        train_online=False,
    )
    groch = ensemble.GroCHClassifier(
        feature_subset=features,
        dist_metric="mahalanobis",
        inv_cov=inv_cov,
        distillator=dist,
        train_after_drift_evaluation=True,
        log_level=0,
    )
    for row, y in zip(X, y_vals):
        x_dict = dict(zip(features, row))
        groch.learn_one(x_dict, str(y))

    arr = groch._active_distillator.prototypes_array
    assert arr.ndim == 2 and arr.shape[1] == 5 and len(arr) >= 2
    d_self = groch._mean_nn_dist(arr, groch._active_distillator)
    assert d_self == 0.0
