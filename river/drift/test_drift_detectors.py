from __future__ import annotations

import numpy as np
import pytest

from river import drift

np.random.seed(12345)
data_stream_1 = np.concatenate((np.random.randint(2, size=1000), np.random.randint(8, size=1000)))

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
    expected_indices = [1055]
    detected_indices = perform_test(drift.ADWIN(), data_stream_1)

    assert detected_indices == expected_indices


def test_ddm():
    expected_indices = [1049]
    detected_indices = perform_test(drift.binary.DDM(), data_stream_2)
    assert detected_indices == expected_indices

    expected_indices = []
    detected_indices = perform_test(drift.binary.DDM(), np.ones(1000))
    assert detected_indices == expected_indices


def test_eddm():
    expected_indices = [1059]
    detected_indices = perform_test(drift.binary.EDDM(alpha=0.9, beta=0.8), data_stream_2)
    assert detected_indices == expected_indices


def test_hddm_a():
    hddm_a = drift.binary.HDDM_A()
    expected_indices = [1047]
    detected_indices = perform_test(hddm_a, data_stream_2)
    assert detected_indices == expected_indices

    # Second test, more abrupt drifts
    hddm_a = drift.binary.HDDM_A(two_sided_test=True)
    expected_indices = [531, 1015, 1545]
    detected_indices = perform_test(hddm_a, data_stream_3)
    assert detected_indices == expected_indices


def test_hddm_w():
    hddm_w = drift.binary.HDDM_W()
    expected_indices = [1018]
    detected_indices = perform_test(hddm_w, data_stream_2)
    assert detected_indices == expected_indices

    # Second test, more abrupt drifts
    hddm_w = drift.binary.HDDM_W(two_sided_test=True)
    expected_indices = [507, 1032, 1508]
    detected_indices = perform_test(hddm_w, data_stream_3)
    assert detected_indices == expected_indices


def test_kswin():
    kswin = drift.KSWIN(alpha=0.0001, window_size=200, stat_size=100, seed=42)
    expected_indices = [1042]
    detected_indices = perform_test(kswin, data_stream_1)
    assert detected_indices == expected_indices


def test_kswin_coverage():
    with pytest.raises(ValueError):
        drift.KSWIN(alpha=-0.1)

    with pytest.raises(ValueError):
        drift.KSWIN(alpha=1.1)

    try:
        drift.KSWIN(window_size=-10)
    except ValueError:
        assert True
    else:
        assert False
    try:
        drift.KSWIN(window_size=10, stat_size=30)
    except ValueError:
        assert True
    else:
        assert False


def test_page_hinkley():
    expected_indices = [588, 1681]
    detected_indices = perform_test(drift.PageHinkley(mode="up"), data_stream_3)

    assert detected_indices == expected_indices

    expected_indices = [1172]
    detected_indices = perform_test(drift.PageHinkley(mode="down"), data_stream_3)

    assert detected_indices == expected_indices

    expected_indices = [588, 1097, 1585]
    detected_indices = perform_test(drift.PageHinkley(mode="both"), data_stream_3)

    assert detected_indices == expected_indices


def perform_test(drift_detector, data_stream):
    detected_indices = []
    for i, val in enumerate(data_stream):
        drift_detector.update(val)
        if drift_detector.drift_detected:
            detected_indices.append(i)
    return detected_indices
