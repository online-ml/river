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
    expected_indices = [1023]
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
    hddm_a = drift.binary.HDDMA()
    expected_indices = [1047]
    detected_indices = perform_test(hddm_a, data_stream_2)
    assert detected_indices == expected_indices

    # Second test, more abrupt drifts
    hddm_a = drift.binary.HDDMA(two_sided_test=True)
    expected_indices = [531, 1015, 1545]
    detected_indices = perform_test(hddm_a, data_stream_3)
    assert detected_indices == expected_indices


def test_hddm_w():
    hddm_w = drift.binary.HDDMW()
    expected_indices = [1018]
    detected_indices = perform_test(hddm_w, data_stream_2)
    assert detected_indices == expected_indices

    # Second test, more abrupt drifts
    hddm_w = drift.binary.HDDMW(two_sided_test=True)
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


def test_shewhart():
    detected_indices = perform_test(drift.Shewhart(), data_stream_1)

    assert len(detected_indices) >= 1
    assert any(1000 <= idx <= 1100 for idx in detected_indices)


def test_shewhart_modes():
    # Shewhart judges each sample against fixed limits, so it needs a continuous stream: on a
    # binary one the center line sits between the two values and nothing can fall outside.
    low = [float(i % 2) * 0.1 for i in range(200)]
    high = [10.0 + (i % 2) * 0.1 for i in range(200)]

    detected_indices = perform_test(drift.Shewhart(mode="up"), low + high)

    assert any(200 <= idx < 400 for idx in detected_indices)
    assert not any(idx < 200 for idx in detected_indices)

    # A one-sided chart only re-baselines when it detects something, so the falling case needs
    # the high regime first: a chart that never saw the rise would still be centred on the low
    # values and would not read the return to them as a drop.
    detected_indices = perform_test(drift.Shewhart(mode="down"), high + low)

    assert any(200 <= idx < 400 for idx in detected_indices)
    assert not any(idx < 200 for idx in detected_indices)


def test_shewhart_coverage():
    with pytest.raises(ValueError):
        drift.Shewhart(k=0)

    with pytest.raises(ValueError):
        drift.Shewhart(k=-1)

    with pytest.raises(ValueError):
        drift.Shewhart(min_instances=0)

    with pytest.raises(ValueError):
        drift.Shewhart(mode="sideways")

    # A constant stream has no spread, so nothing can fall outside the limits.
    detector = drift.Shewhart(min_instances=3)

    for _ in range(100):
        detector.update(1.0)

    assert detector.drift_detected is False

    # Nothing is flagged before the limits exist.
    detector = drift.Shewhart(min_instances=50, k=0.5)

    for _ in range(40):
        detector.update(100.0)

    assert detector.drift_detected is False

    # A point well outside the frozen limits is flagged.
    detector = drift.Shewhart(min_instances=5, k=3.0)

    for _ in range(20):
        detector.update(0.0)
    for _ in range(5):
        detector.update(0.0)
        assert detector.drift_detected is False

    for _ in range(50):
        detector.update(100.0)
        if detector.drift_detected:
            break

    assert detector.drift_detected is True
