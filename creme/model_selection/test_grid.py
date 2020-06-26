import pytest

from creme import compose
from creme import linear_model
from creme import model_selection
from creme import optim
from creme import preprocessing
from creme import tree


@pytest.mark.parametrize('param_grid, count', [
    (
        {
            'LinearRegression': {
                'optimizer': [
                    (optim.SGD, {'lr': [1, 2]}),
                    (optim.Adam, {
                        'beta_1': [0.1, 0.01, 0.001],
                        'lr': [0.1, 0.01, 0.001, 0.0001]
                    })
                ]
            }
        },
        2 + 3 * 4
    ),
    (
        {
            'Scaler': [
                preprocessing.MinMaxScaler(),
                preprocessing.MaxAbsScaler(),
                preprocessing.StandardScaler()
            ],
            'LinearRegression': {
                'optimizer': {
                    'lr': [1e-1, 1e-2, 1e-3]
                }
            }
        },
        3 * 3
    )

])
def test_expand_param_grid_count(param_grid, count):
    assert sum(1 for _ in model_selection.expand_param_grid(param_grid)) == count


def test_decision_tree_max_depth():

    model = tree.DecisionTreeClassifier()

    max_depths = [1, 2, 3, 4, 5, 6]
    param_grid = model_selection.expand_param_grid({
        'max_depth': max_depths
    })

    models = [model._set_params(params) for params in param_grid]

    for model, max_depth in zip(models, max_depths):
        assert model.max_depth == max_depth
