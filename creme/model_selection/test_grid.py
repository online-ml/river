import pytest

from creme import model_selection
from creme import optim


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
    )
])
def test_expand_param_grid_count(param_grid, count):
    assert sum(1 for _ in model_selection.expand_param_grid(param_grid)) == count
