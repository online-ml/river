from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import neighbors

from river import anomaly, datasets

np.random.seed(42)


def test_incremental_lof_scores():
    """
    Test that the incremental LOF algorithm returns similar LOF scores for each observation
    compared with the original static LOF algorithm implemented in scikit-learn.
    """
    norm_dist = 0.5 * np.random.rand(100, 2)
    x_inliers = np.concatenate((norm_dist - 2, norm_dist, norm_dist + 2), axis=0)
    x_outliers = np.concatenate(
        (
            np.random.uniform(low=-4, high=4, size=(20, 2)),
            np.random.uniform(low=-10, high=-5, size=(10, 2)),
            np.random.uniform(low=5, high=10, size=(10, 2)),
        ),
        axis=0,
    )
    x_train = np.concatenate((x_inliers, x_outliers), axis=0)
    x_train_dict = [{f"feature_{i + 1}": elem[i] for i in range(2)} for elem in x_train]
    ground_truth = np.ones(len(x_train), dtype=int)
    ground_truth[-len(x_outliers) :] = -1
    df_train = pd.DataFrame({"observations": x_train_dict, "ground_truth": ground_truth})
    x_pred = np.random.uniform(low=-5, high=5, size=(30, 2))
    x_pred_dict = [{f"feature_{i + 1}": elem[i] for i in range(2)} for elem in x_pred]
    incremental_lof = anomaly.LocalOutlierFactor(n_neighbors=20)

    for x in df_train["observations"]:
        incremental_lof.learn_one(x)

    ilof_scores_train = np.array([ilof_score for ilof_score in incremental_lof.lof.values()])

    ilof_scores_pred = []
    for x in x_pred_dict:
        ilof_scores_pred.append(incremental_lof.score_one(x))

    lof_sklearn = neighbors.LocalOutlierFactor(n_neighbors=20)
    lof_sklearn.fit_predict(x_train)
    lof_sklearn_scores_train = -lof_sklearn.negative_outlier_factor_

    assert np.allclose(ilof_scores_train, lof_sklearn_scores_train, rtol=1e-08, atol=1e-08)


def test_batch_lof_scores():
    """
    Test that the incremental LOF algorithm returns similar LOF scores for each batch
    with `learn_many` compared with the original static LOF algorithm implemented in scikit-learn,
    under different batch sizes.
    """
    cc_df = pd.DataFrame(datasets.CreditCard())
    cc_df_np = [np.array(list(x.values())) for x in cc_df[0].to_dict().values()]

    batch_sizes = [20, 50, 100]

    for batch_size in batch_sizes:
        ilof_river_batch = anomaly.LocalOutlierFactor(n_neighbors=20)
        ilof_river_batch.learn_many(cc_df[0:batch_size])
        ilof_scores_river_batch = np.array([v for v in ilof_river_batch.lof.values()])

        lof_sklearn_batch = neighbors.LocalOutlierFactor(n_neighbors=20)
        lof_sklearn_batch.fit_predict(cc_df_np[0:batch_size])
        lof_scores_sklearn_batch = -lof_sklearn_batch.negative_outlier_factor_

        assert np.allclose(
            ilof_scores_river_batch, lof_scores_sklearn_batch, rtol=1e-02, atol=1e-02
        )


def test_issue_1328():
    lof = anomaly.LocalOutlierFactor()
    X = [{"a": 1, "b": 1}, {"a": 1, "b": 1}]
    for x in X:
        lof.learn_one(x)


def test_issue_1331():
    import copy

    from river import anomaly

    lof = anomaly.LocalOutlierFactor()

    X = [{"a": 1, "b": 1}, {"a": 1, "b": 1}]
    for x in X:
        lof.learn_one(x)

    neighborhoods_ = lof.neighborhoods.copy()
    rev_neighborhoods = lof.rev_neighborhoods.copy()
    k_dist_ = lof.k_dist.copy()
    reach_dist_ = copy.deepcopy(lof.reach_dist)
    dist_dict_ = copy.deepcopy(lof.dist_dict)
    local_reach_ = lof.local_reach.copy()
    lof_ = lof.lof.copy()

    lof.score_one({"a": 0.5, "b": 1})

    assert neighborhoods_ == lof.neighborhoods
    assert rev_neighborhoods == lof.rev_neighborhoods
    assert k_dist_ == lof.k_dist
    assert reach_dist_ == lof.reach_dist
    assert dist_dict_ == lof.dist_dict
    assert local_reach_ == lof.local_reach
    assert lof_ == lof.lof
