import pytest

from river import compose
from river import linear_model
from river import optim
from river import preprocessing
from river import tree
from river import utils


@pytest.mark.parametrize(
    "model, param_grid, count",
    [
        (
            linear_model.LinearRegression(),
            {
                "optimizer": [
                    (optim.SGD, {"lr": [1, 2]}),
                    (
                        optim.Adam,
                        {
                            "beta_1": [0.1, 0.01, 0.001],
                            "lr": [0.1, 0.01, 0.001, 0.0001],
                        },
                    ),
                ]
            },
            2 + 3 * 4,
        ),
        (
            preprocessing.StandardScaler() | linear_model.LinearRegression(),
            {
                "LinearRegression": {
                    "optimizer": [
                        (optim.SGD, {"lr": [1, 2]}),
                        (
                            optim.Adam,
                            {
                                "beta_1": [0.1, 0.01, 0.001],
                                "lr": [0.1, 0.01, 0.001, 0.0001],
                            },
                        ),
                    ]
                }
            },
            2 + 3 * 4,
        ),
        (
            compose.Pipeline(("Scaler", None), linear_model.LinearRegression()),
            {
                "Scaler": [
                    preprocessing.MinMaxScaler(),
                    preprocessing.MaxAbsScaler(),
                    preprocessing.StandardScaler(),
                ],
                "LinearRegression": {"optimizer": {"lr": [1e-1, 1e-2, 1e-3]}},
            },
            3 * 3,
        ),
    ],
)
def test_expand_param_grid_count(model, param_grid, count):
    assert len(utils.expand_param_grid(model, param_grid)) == count


def test_decision_tree_max_depth():

    model = tree.HoeffdingTreeClassifier()

    max_depths = [1, 2, 3, 4, 5, 6]
    models = utils.expand_param_grid(model, {"max_depth": max_depths})

    for model, max_depth in zip(models, max_depths):
        assert model.max_depth == max_depth
