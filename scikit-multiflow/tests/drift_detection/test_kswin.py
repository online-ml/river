from river.drift import KSWIN
import numpy as np
import pytest
import os

def test_kswin_initialization():
    """
    KSWIN Test

    Content:
    alpha initialisation test.
    alpha has range from (0,1)

    pre obtained data initialisation test.
    data must be list

    KSWIN window size initialisation test.
    0 < stat_size <  window_size

    KSWIN change detector size initialisation test.
    At least 1 false positive must arisie due to the sensitive alpha, when testing the standard
    Sea generator
    """
    with pytest.raises(ValueError):
        KSWIN(alpha=-0.1)

    with pytest.raises(ValueError):
        KSWIN(alpha=1.1)

    kswin = KSWIN(alpha=0.5)
    assert kswin.alpha == 0.5


    kswin = KSWIN(data="st")
    assert isinstance(kswin.window, np.ndarray)


    kswin = KSWIN(data=np.array([0.75,0.80,1,-1]))
    assert isinstance(kswin.window, np.ndarray)

    try:
        KSWIN(window_size=-10)
    except ValueError:
        assert True
    else:
        assert False
    try:
        KSWIN(window_size=10, stat_size=30)
    except ValueError:
        assert True
    else:
        assert False


def test_kswin_functionality(test_path):
    kswin = KSWIN(alpha=0.0001,window_size=200,stat_size=100)
    test_file = os.path.join(test_path, 'drift_stream.npy')
    data_stream = np.load(test_file)
    expected_indices = [1045, 1145]
    detected_indices = []

    for i in range(data_stream.size):
        kswin.update(data_stream[i])
        if kswin.change_detected:
            detected_indices.append(i)

    assert detected_indices == expected_indices


def test_kswin_reset():
    kswin = KSWIN()
    kswin.reset()
    assert kswin.p_value == 0
    assert kswin.window.shape[0] == 0
    assert kswin.change_detected == False
