import pytest

from creme.utils import math


@pytest.mark.parametrize(
    'dict_pair',
    [
        (({'x0': 1, 'x1': 2}, {'x1': 21, 'x2': 3}), 42),
        (({'x0': 1, 'x1': 2}, {'x0': 23}), 23),
        (({'x0': 2, 'x1': 3}, {'y0': 5, 'y1': 7}), 0)
    ]
)
def test_dot(dict_pair, value):
    a, b = dict_pair
    assert math.dot(a, b) == value
    assert math.dot(b, a) == value
