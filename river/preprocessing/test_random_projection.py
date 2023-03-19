import numpy as np
from river import datasets
from river import preprocessing


def test_dot_product():
    dataset = datasets.TrumpApproval()
    model = preprocessing.GaussianRandomProjector(n_components=3, seed=42)

    for x, y in dataset:
        y = model.transform_one(x)
        y_arr = np.array(list(y.values()))
        x_arr = np.array(list(x.values()))
        P = np.array(
            [[model._projection_matrix[i, j] for j in x] for i in range(model.n_components)]
        )
        np.testing.assert_allclose(x_arr @ P.T, y_arr)
