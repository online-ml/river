import numpy as np
import pytest
from skmultiflow.utils import check_random_state
from skmultiflow.utils import check_weights


def test_check_random_state():
    rand = None

    rand = check_random_state(rand)
    assert isinstance(rand, np.random.mtrand.RandomState)

    rand = check_random_state(rand)
    assert isinstance(rand, np.random.mtrand.RandomState)

    rand = check_random_state(int(1))
    assert isinstance(rand, np.random.mtrand.RandomState)

    with pytest.raises(ValueError):
        check_random_state(2.0)


def test_check_weights():
    test_value = 5
    weights = check_weights(int(test_value))
    assert weights == int(test_value)

    weights = check_weights(float(test_value))
    assert weights == float(test_value)

    weights = check_weights(np.float(test_value))
    assert weights == np.float(test_value)

    weights = check_weights(np.int(test_value))
    assert weights == np.int(test_value)

    weights = check_weights(np.array([test_value], np.float))
    assert isinstance(weights, np.ndarray)
    assert isinstance(weights[0], np.float)

    weights = check_weights(np.array(np.arange(test_value)))
    assert isinstance(weights, np.ndarray)
    for w in weights:
        assert isinstance(w, np.integer)

    weights = check_weights([float(x) for x in range(test_value)])
    assert isinstance(weights, list)
    for w in weights:
        assert isinstance(w, float)

    weights = check_weights(int(test_value), expand_length=10)
    assert isinstance(weights, np.ndarray)
    for w in weights:
        assert w == int(test_value)

    with pytest.raises(ValueError):
        check_weights([1.0, 2.0, 3.0, 4.0, 'invalid'])

    with pytest.raises(ValueError):
        check_weights('invalid')

    with pytest.raises(ValueError):
        check_weights([float(x) for x in range(test_value)], expand_length=10)

    with pytest.raises(ValueError):
        check_weights(int(test_value), expand_length=-10)
