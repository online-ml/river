import numpy as np
import pytest

from river.drift import ADWIN, DDM, EDDM, HDDM_A, HDDM_W, KSWIN, PageHinkley

np.random.seed(12345)
data_stream_1 = np.concatenate(
    (np.random.randint(2, size=1000), np.random.randint(8, size=1000))
)

np.random.seed(12345)
data_stream_2 = np.concatenate(
    [
        [np.random.binomial(1, 0.2) for _ in range(1000)],
        [np.random.binomial(1, 0.8) for _ in range(1000)],
    ]
).astype(int)

np.random.seed(12345)
data_stream_3 = np.concatenate(
    (
        np.random.normal(0.0, 0.1, 500) > 0,
        np.random.normal(0.25, 0.1, 500) > 0,
        np.random.normal(0.0, 0.1, 500) > 0,
        np.random.normal(0.25, 0.1, 500) > 0,
    )
).astype(int)


def test_adwin():
    expected_indices = [1055, 1087, 1215]
    detected_indices = perform_test(ADWIN(), data_stream_1)

    assert detected_indices == expected_indices


def test_ddm():
    expected_indices = [1049]
    detected_indices = perform_test(DDM(), data_stream_2)
    assert detected_indices == expected_indices


def test_eddm():
    expected_indices = [681, 848, 983, 1042, 1089]
    detected_indices = perform_test(EDDM(), data_stream_2)
    assert detected_indices == expected_indices


def test_hddm_a():
    hddm_a = HDDM_A()
    expected_indices = [1047]
    detected_indices = perform_test(hddm_a, data_stream_2)
    assert detected_indices == expected_indices

    # Second test, more abrupt drifts
    hddm_a = HDDM_A(two_sided_test=True)
    expected_indices = [531, 1545]
    detected_indices = perform_test(hddm_a, data_stream_3)
    assert detected_indices == expected_indices


def test_hddm_w():
    hddm_w = HDDM_W()
    expected_indices = [1018]
    detected_indices = perform_test(hddm_w, data_stream_2)
    assert detected_indices == expected_indices

    # Second test, more abrupt drifts
    hddm_w = HDDM_W(two_sided_test=True)
    expected_indices = [507, 1508]
    detected_indices = perform_test(hddm_w, data_stream_3)
    assert detected_indices == expected_indices


def test_kswin():
    kswin = KSWIN(alpha=0.0001, window_size=200, stat_size=100)
    expected_indices = [1043, 1143]
    detected_indices = perform_test(kswin, data_stream_1)
    assert detected_indices == expected_indices


def test_kswin_coverage():
    with pytest.raises(ValueError):
        KSWIN(alpha=-0.1)

    with pytest.raises(ValueError):
        KSWIN(alpha=1.1)

    kswin = KSWIN(alpha=0.5)
    assert kswin.alpha == 0.5

    kswin = KSWIN(window="st")
    assert isinstance(kswin.window, np.ndarray)

    kswin = KSWIN(window=np.array([0.75, 0.80, 1, -1]))
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

    kswin = KSWIN()
    kswin.reset()
    assert kswin.p_value == 0
    assert kswin.window.shape[0] == 0
    assert kswin.change_detected is False


def test_page_hinkley():
    expected_indices = [1020, 1991]
    detected_indices = perform_test(PageHinkley(), data_stream_1)

    assert detected_indices == expected_indices


def perform_test(drift_detector, data_stream):
    detected_indices = []
    for i, val in enumerate(data_stream):
        in_drift, in_warning = drift_detector.update(val)
        if in_drift:
            detected_indices.append(i)
    return detected_indices
