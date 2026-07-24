from __future__ import annotations

import math

import numpy as np
import pytest

from river import drift


def test_alarm_inside_window_is_a_true_positive():
    matching = drift.evaluate.match(alarms=[120], drifts=[100], delta_max=50)

    assert matching.hits == {0: [120]}
    assert matching.false_alarms == []


def test_alarm_after_window_is_a_false_alarm():
    matching = drift.evaluate.match(alarms=[151], drifts=[100], delta_max=50)

    assert matching.hits == {}
    assert matching.false_alarms == [151]


def test_window_upper_bound_is_inclusive():
    matching = drift.evaluate.match(alarms=[150], drifts=[100], delta_max=50)

    assert matching.hits == {0: [150]}


def test_alarm_before_drift_is_a_false_alarm_by_default():
    matching = drift.evaluate.match(alarms=[99], drifts=[100], delta_max=50)

    assert matching.false_alarms == [99]


def test_delta_pre_allows_early_detection():
    matching = drift.evaluate.match(alarms=[95], drifts=[100], delta_max=50, delta_pre=10)

    assert matching.hits == {0: [95]}


def test_gradual_drift_is_given_as_an_interval():
    matching = drift.evaluate.match(alarms=[250], drifts=[(100, 200)], delta_max=50)

    assert matching.hits == {0: [250]}


def test_alarm_past_the_end_of_a_gradual_drift_window_is_a_false_alarm():
    matching = drift.evaluate.match(alarms=[251], drifts=[(100, 200)], delta_max=50)

    assert matching.false_alarms == [251]


def test_an_alarm_is_matched_to_the_earliest_open_window():
    matching = drift.evaluate.match(alarms=[130], drifts=[100, 120], delta_max=50)

    assert matching.hits == {0: [130]}


def test_recall_counts_every_alarm_in_a_window():
    recall = drift.evaluate.recall(alarms=[110, 130], drifts=[100, 500], delta_max=50)

    assert recall == pytest.approx(2 / 3)


def test_episode_recall_counts_a_window_once():
    recall = drift.evaluate.episode_recall(alarms=[110, 130], drifts=[100, 500], delta_max=50)

    assert recall == 0.5


def test_episode_recall_is_the_fraction_of_drifts_caught():
    assert drift.evaluate.episode_recall(alarms=[110], drifts=[100, 500], delta_max=50) == 0.5


def test_missed_detection_rate_is_one_minus_episode_recall():
    assert (
        drift.evaluate.missed_detection_rate(alarms=[110], drifts=[100, 500], delta_max=50) == 0.5
    )


def test_precision_is_the_fraction_of_alarms_that_landed():
    assert drift.evaluate.precision(alarms=[110, 900], drifts=[100], delta_max=50) == 0.5


def test_precision_is_undefined_without_alarms():
    assert math.isnan(drift.evaluate.precision(alarms=[], drifts=[100], delta_max=50))


def test_f1_is_the_harmonic_mean_of_precision_and_episode_recall():
    f1 = drift.evaluate.f1(alarms=[110, 900], drifts=[100, 500], delta_max=50)

    assert f1 == pytest.approx(2 * 0.5 * 0.5 / (0.5 + 0.5))


def test_detection_delay_measures_from_the_start_of_the_drift():
    assert drift.evaluate.detection_delay(alarms=[110, 130], drifts=[100], delta_max=50) == 10.0


def test_detection_delay_averages_over_caught_drifts_only():
    delay = drift.evaluate.detection_delay(alarms=[110, 530], drifts=[100, 500, 900], delta_max=50)

    assert delay == 20.0


def test_detection_delay_is_undefined_when_nothing_is_caught():
    assert math.isnan(drift.evaluate.detection_delay(alarms=[], drifts=[100], delta_max=50))


def test_normalized_detection_time_divides_the_delay_by_delta_max():
    ndt = drift.evaluate.normalized_detection_time(alarms=[110], drifts=[100], delta_max=50)

    assert ndt == pytest.approx(0.2)


def test_false_alarm_rate_is_per_sample():
    far = drift.evaluate.false_alarm_rate(
        alarms=[110, 900], drifts=[100], n_samples=1000, delta_max=50
    )

    assert far == pytest.approx(0.001)


def test_false_alarm_rate_can_be_scaled_to_another_time_unit():
    far = drift.evaluate.false_alarm_rate(
        alarms=[110, 900], drifts=[100], n_samples=1000, delta_max=50, unit=1000
    )

    assert far == pytest.approx(1.0)


def test_mean_time_between_false_alarms_averages_the_gaps():
    mtfa = drift.evaluate.mean_time_between_false_alarms(
        alarms=[200, 300, 500], drifts=[], delta_max=50
    )

    assert mtfa == pytest.approx(150.0)


def test_mean_time_between_false_alarms_is_undefined_below_two_false_alarms():
    mtfa = drift.evaluate.mean_time_between_false_alarms(alarms=[200], drifts=[], delta_max=50)

    assert math.isnan(mtfa)


def test_mean_time_ratio_combines_the_three_ground_truth_metrics():
    alarms = [110, 400, 700]
    drifts = [100, 900]

    mtr = drift.evaluate.mean_time_ratio(alarms=alarms, drifts=drifts, delta_max=50)

    assert mtr == pytest.approx((300.0 / 10.0) * (1 - 0.5))


def test_mean_time_ratio_is_undefined_below_two_false_alarms():
    mtr = drift.evaluate.mean_time_ratio(alarms=[110], drifts=[100], delta_max=50)

    assert math.isnan(mtr)


def test_report_collects_every_metric():
    report = drift.evaluate.report(
        alarms=[110, 900], drifts=[100, 500], n_samples=1000, delta_max=50
    )

    assert report["episode_recall"] == 0.5
    assert report["precision"] == 0.5
    assert report["missed_detection_rate"] == 0.5
    assert report["detection_delay"] == 10.0
    assert report["false_alarm_rate"] == pytest.approx(0.001)
    assert report["n_alarms"] == 2


def test_score_drives_a_detector_over_a_stream():
    rng = np.random.RandomState(12345)
    stream = np.concatenate((rng.normal(0, 0.1, 1000), rng.normal(5, 0.1, 1000)))

    report = drift.evaluate.score(drift.ADWIN(), stream, drifts=[1000], delta_max=200)

    assert report["episode_recall"] == 1.0
    assert 0 <= report["detection_delay"] <= 200
    assert report["n_samples"] == 2000


def test_score_reports_a_miss_when_the_detector_stays_quiet():
    stream = [0.0] * 2000

    report = drift.evaluate.score(drift.ADWIN(), stream, drifts=[1000], delta_max=200)

    assert report["episode_recall"] == 0.0
    assert report["missed_detection_rate"] == 1.0
    assert report["n_alarms"] == 0


def test_delta_max_must_be_positive():
    with pytest.raises(ValueError):
        drift.evaluate.match(alarms=[110], drifts=[100], delta_max=0)


def test_drift_intervals_must_be_ordered():
    with pytest.raises(ValueError):
        drift.evaluate.match(alarms=[110], drifts=[(200, 100)], delta_max=50)
