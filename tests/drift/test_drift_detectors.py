from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from river import checks, drift

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


def test_wstd():
    expected_indices = [1028]
    detected_indices = perform_test(drift.binary.WSTD(), data_stream_2)
    assert detected_indices == expected_indices

    wstd = drift.binary.WSTD()
    warning_indices = []
    for i, x in enumerate(data_stream_2):
        wstd.update(x)
        if wstd.warning_detected:
            warning_indices.append(i)
    assert warning_indices[0] == 1017


def test_wstd_stationary_no_false_alarm():
    np.random.seed(12345)
    stationary = np.array([np.random.binomial(1, 0.2) for _ in range(2000)])
    assert perform_test(drift.binary.WSTD(), stationary) == []

    np.random.seed(12345)
    stationary = np.random.randint(2, size=2000)
    assert perform_test(drift.binary.WSTD(), stationary) == []

    assert perform_test(drift.binary.WSTD(), np.ones(1000)) == []
    assert perform_test(drift.binary.WSTD(), np.zeros(1000)) == []


def test_wstd_two_sided():
    data_stream = np.array([0] * 200 + [1] * 60 + [0] * 200)
    expected_indices = [210, 273]
    assert perform_test(drift.binary.WSTD(), data_stream) == expected_indices


def test_wstd_p_value_matches_ranksums():
    rng = np.random.default_rng(42)
    data_stream = rng.integers(0, 2, size=400)

    wstd = drift.binary.WSTD()
    history = []
    for i, x in enumerate(data_stream):
        wstd.update(x)
        history.append(x)
        if i >= wstd.recent_window_size + wstd.older_window_size - 1:
            recent = history[-wstd.recent_window_size :]
            older = history[
                -wstd.older_window_size - wstd.recent_window_size : -wstd.recent_window_size
            ]
            _, p_value = stats.ranksums(recent, older)
            assert math.isclose(wstd.p_value, p_value, rel_tol=1e-12, abs_tol=1e-300)


def test_wstd_guard():
    wstd = drift.binary.WSTD(recent_window_size=10, older_window_size=20, min_instances=10)
    for x in [0] * 19:
        wstd.update(x)
        assert wstd.p_value == 1.0
    wstd.update(0)
    assert wstd.p_value == 1.0


def test_wstd_coverage():
    with pytest.raises(ValueError):
        drift.binary.WSTD(alpha_warning=0.001, alpha_drift=0.003)

    with pytest.raises(ValueError):
        drift.binary.WSTD(alpha_warning=-0.1)

    with pytest.raises(ValueError):
        drift.binary.WSTD(alpha_drift=1.1)

    with pytest.raises(ValueError):
        drift.binary.WSTD(recent_window_size=1)

    with pytest.raises(ValueError):
        drift.binary.WSTD(min_instances=200)


def test_wstd_check_estimator():
    wstd = drift.binary.WSTD()
    for check in [
        checks.common.check_repr,
        checks.common.check_str,
        checks.common.check_doc,
        checks.common.check_clone_same_class,
        checks.common.check_clone_is_idempotent,
        checks.common.check_init_has_default_params_for_tests,
        checks.common.check_init_default_params_are_not_mutable,
        checks.common.check_clone_changes_memory_addresses,
        checks.common.check_mutate_can_be_idempotent,
        checks.common.check_pickling_supports_roundtrip,
        checks.common.check_repr_roundtrips_clone,
        checks.common.check_clone_with_new_params_applies,
        checks.common.check_get_params_matches_signature,
    ]:
        check(wstd.clone())


def perform_test(drift_detector, data_stream):
    detected_indices = []
    for i, val in enumerate(data_stream):
        drift_detector.update(val)
        if drift_detector.drift_detected:
            detected_indices.append(i)
    return detected_indices
