from __future__ import annotations

import numpy as np

from river import datasets, preprocessing


def test_gaussian_random_projector_dot_product():
    dataset = datasets.TrumpApproval()
    projector = preprocessing.GaussianRandomProjector(n_components=3)

    for x, y in dataset:
        y = projector.transform_one(x)
        y_arr = np.array(list(y.values()))
        x_arr = np.array(list(x.values()))
        P = np.array(
            [[projector._projection_matrix[i, j] for j in x] for i in range(projector.n_components)]
        )
        np.testing.assert_allclose(x_arr @ P.T, y_arr)


def test_sparse_random_projector_dot_product():
    dataset = datasets.TrumpApproval()
    projector = preprocessing.SparseRandomProjector(n_components=3, density=0.5)

    for x, y in dataset:
        y = projector.transform_one(x)
        y_arr = np.array(list(y.values()))
        x_arr = np.array(list(x.values()))
        P = np.array(
            [
                [projector._projection_matrix[j].get(i, 0) for j in x]
                for i in range(projector.n_components)
            ]
        )
        np.testing.assert_allclose(x_arr @ P.T, y_arr)


def test_sparse_random_projector_size():
    dataset = datasets.TrumpApproval()
    projector = preprocessing.SparseRandomProjector(n_components=3, density=0.5)

    for x, y in dataset:
        projector.transform_one(x)
        break

    n_weights = sum(len(v) for v in projector._projection_matrix.values())
    assert n_weights < len(x) * projector.n_components
