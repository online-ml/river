"""Comprehensive regression tests for ADWIN to guard against optimization regressions."""

from __future__ import annotations

import random

import numpy as np
import pytest

from river import drift
from river.drift.adwin_c import AdaptiveWindowing

# ---- Drift detection correctness ----


class TestDriftDetection:
    """Tests that drift detection results are numerically identical."""

    def test_basic_drift_at_expected_index(self):
        """The canonical test from the existing suite."""
        np.random.seed(12345)
        data = np.concatenate((np.random.randint(2, size=1000), np.random.randint(8, size=1000)))
        adwin = drift.ADWIN()
        detected = []
        for i, val in enumerate(data):
            adwin.update(val)
            if adwin.drift_detected:
                detected.append(i)
        assert detected == [1023]

    def test_multiple_drifts(self):
        """Detect multiple drifts in a multi-segment stream."""
        rng = random.Random(42)
        data = (
            rng.choices([0, 1], k=1000)
            + rng.choices(range(4, 8), k=1000)
            + rng.choices([0, 1], k=1000)
            + rng.choices(range(4, 8), k=1000)
        )
        adwin = drift.ADWIN()
        detected = []
        for i, val in enumerate(data):
            adwin.update(val)
            if adwin.drift_detected:
                detected.append(i)
        # Record expected behavior
        assert len(detected) >= 2, f"Expected multiple drifts, got {detected}"
        # Exact expected indices (captured from current implementation)
        assert detected == [1023, 2015, 3007]

    def test_no_drift_constant_stream(self):
        """No drift should be detected in a constant stream."""
        adwin = drift.ADWIN()
        for _ in range(5000):
            adwin.update(1.0)
            assert not adwin.drift_detected

    def test_no_drift_stable_distribution(self):
        """No drift in a stationary random stream."""
        rng = random.Random(123)
        adwin = drift.ADWIN()
        for _ in range(5000):
            adwin.update(rng.gauss(0, 1))
        # Might detect a false positive or two, but shouldn't detect many
        assert adwin.n_detections <= 2

    def test_immediate_large_shift(self):
        """Large distribution shift should be detected quickly."""
        adwin = drift.ADWIN(delta=0.01)
        for _ in range(200):
            adwin.update(0.0)
        detected = False
        for i in range(200):
            adwin.update(100.0)
            if adwin.drift_detected:
                detected = True
                break
        assert detected, "Large shift should be detected"


# ---- Statistics correctness ----


class TestStatistics:
    """Tests that ADWIN statistics (width, total, variance, estimation) are correct."""

    def test_width_grows(self):
        """Width should grow as elements are added (before any drift)."""
        adwin = drift.ADWIN()
        for i in range(1, 101):
            adwin.update(1.0)
            assert adwin.width == i

    def test_total_and_estimation(self):
        """Total and estimation should be consistent."""
        adwin = drift.ADWIN()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            adwin.update(v)
        assert adwin.total == pytest.approx(15.0)
        assert adwin.estimation == pytest.approx(3.0)
        assert adwin.width == 5

    def test_variance_correctness(self):
        """Variance should match the online Welford computation."""
        rng = random.Random(999)
        adwin = drift.ADWIN()
        values = [rng.gauss(5, 2) for _ in range(500)]
        for v in values:
            adwin.update(v)
        # ADWIN stores sum of squared deviations (not divided by n)
        expected_var = sum((v - (sum(values) / len(values))) ** 2 for v in values)
        assert adwin.variance == pytest.approx(expected_var, rel=1e-6)

    def test_estimation_after_drift(self):
        """After drift, estimation should reflect the new window only."""
        adwin = drift.ADWIN()
        for _ in range(1000):
            adwin.update(0.0)
        for _ in range(1000):
            adwin.update(10.0)
        # After drift is detected and old window is dropped,
        # estimation should be close to 10.0
        assert adwin.estimation > 5.0

    def test_width_zero_estimation(self):
        """Estimation on empty ADWIN should return 0."""
        adwin = drift.ADWIN()
        assert adwin.estimation == 0.0


# ---- Parameter variations ----


class TestParameters:
    """Tests that different parameter settings work correctly."""

    def test_clock_1(self):
        """clock=1 should check for drift on every update."""
        np.random.seed(12345)
        data = np.concatenate((np.random.randint(2, size=1000), np.random.randint(8, size=1000)))
        adwin = drift.ADWIN(clock=1)
        detected = []
        for i, val in enumerate(data):
            adwin.update(val)
            if adwin.drift_detected:
                detected.append(i)
        assert len(detected) >= 1

    def test_different_deltas(self):
        """Smaller delta should be less sensitive (fewer detections)."""
        np.random.seed(42)
        data = np.concatenate((np.random.randn(500), np.random.randn(500) + 0.5))
        detections = {}
        for delta in [0.2, 0.01, 0.001]:
            adwin = drift.ADWIN(delta=delta)
            count = 0
            for val in data:
                adwin.update(val)
                if adwin.drift_detected:
                    count += 1
            detections[delta] = count
        # More permissive delta should detect at least as many drifts
        assert detections[0.2] >= detections[0.001]

    def test_max_buckets_variation(self):
        """Different max_buckets values should work without error."""
        for mb in [2, 5, 10, 20]:
            adwin = drift.ADWIN(max_buckets=mb)
            rng = random.Random(42)
            for _ in range(2000):
                adwin.update(rng.random())
            assert adwin.width > 0

    def test_min_window_length(self):
        """min_window_length should affect detection sensitivity."""
        rng = random.Random(42)
        adwin = drift.ADWIN(min_window_length=1, clock=1)
        detections = 0
        for _ in range(500):
            adwin.update(rng.gauss(0, 1))
            if adwin.drift_detected:
                detections += 1
        for _ in range(500):
            adwin.update(rng.gauss(5, 1))
            if adwin.drift_detected:
                detections += 1
        # Should detect the change with enough data
        assert detections >= 1

    def test_grace_period(self):
        """No detection should happen within the grace period."""
        adwin = drift.ADWIN(grace_period=100, clock=1)
        for i in range(50):
            adwin.update(0.0)
        for i in range(50):
            adwin.update(1000.0)
        # Within grace period of 100, no detection should happen
        assert adwin.n_detections == 0


# ---- AdaptiveWindowing direct tests ----


class TestAdaptiveWindowing:
    """Direct tests on the Cython AdaptiveWindowing class."""

    def test_basic_update(self):
        aw = AdaptiveWindowing()
        assert not aw.update(1.0)
        assert aw.get_width() == 1.0
        assert aw.get_total() == 1.0

    def test_variance_in_window(self):
        aw = AdaptiveWindowing()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            aw.update(v)
        # variance_in_window = variance / width
        assert aw.variance_in_window == pytest.approx(aw.get_variance() / aw.get_width())

    def test_n_detections_increments(self):
        """n_detections should increment on each detected drift."""
        aw = AdaptiveWindowing(delta=0.01, clock=1)
        for _ in range(500):
            aw.update(0.0)
        for _ in range(500):
            aw.update(100.0)
        assert aw.get_n_detections() >= 1

    def test_sequence_exact_statistics(self):
        """Record exact statistics for a deterministic sequence."""
        aw = AdaptiveWindowing(clock=1000000)  # Never check for drift
        values = [float(i % 7) for i in range(200)]
        for v in values:
            aw.update(v)
        assert aw.get_width() == 200.0
        assert aw.get_total() == pytest.approx(sum(values))
        expected_var = sum((v - sum(values) / 200) ** 2 for v in values)
        assert aw.get_variance() == pytest.approx(expected_var, rel=1e-6)


# ---- Edge cases ----


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_element(self):
        adwin = drift.ADWIN()
        adwin.update(42.0)
        assert adwin.width == 1
        assert adwin.total == 42.0
        assert adwin.estimation == 42.0

    def test_two_elements(self):
        adwin = drift.ADWIN()
        adwin.update(1.0)
        adwin.update(2.0)
        assert adwin.width == 2
        assert adwin.total == 3.0
        assert adwin.estimation == 1.5

    def test_large_values(self):
        adwin = drift.ADWIN()
        for _ in range(100):
            adwin.update(1e15)
        assert adwin.width == 100
        assert adwin.estimation == pytest.approx(1e15)

    def test_negative_values(self):
        adwin = drift.ADWIN()
        for _ in range(100):
            adwin.update(-5.0)
        assert adwin.estimation == pytest.approx(-5.0)

    def test_alternating_values(self):
        """Rapidly alternating values shouldn't cause false drift."""
        adwin = drift.ADWIN()
        for i in range(2000):
            adwin.update(float(i % 2))
        # Stationary alternating pattern shouldn't trigger many drifts
        assert adwin.n_detections <= 1

    def test_very_long_stream(self):
        """ADWIN should handle a long stream without errors or memory issues."""
        rng = random.Random(42)
        adwin = drift.ADWIN()
        for _ in range(100_000):
            adwin.update(rng.random())
        assert adwin.width > 0
        assert 0.0 < adwin.estimation < 1.0

    def test_reset_on_drift_detected(self):
        """After drift_detected is True, next update should reset."""
        adwin = drift.ADWIN(clock=1)
        # Feed data to trigger drift
        for _ in range(500):
            adwin.update(0.0)
        drift_found = False
        for _ in range(500):
            adwin.update(100.0)
            if adwin.drift_detected:
                drift_found = True
                break
        assert drift_found
        # The next update should reset the detector
        adwin.update(50.0)
        # After reset, width should be 1 (fresh start)
        assert adwin.width == 1
