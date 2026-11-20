import pytest

from river.drift.event_benchmark import collect_alarms, score_events

def test_early_duplicate_and_missed():
    result = score_events([100,200], [90,110,111,245], 20)

    assert result["matches"] == [(100, 110)]
    assert result["matched"] == 1
    assert result["missed"] == 1
    assert result["false_alarms"] == 3
    assert result["mean_delay"] == 10

def test_oldest_first():
    result = score_events([100,120], [125], 30)

    assert result["matches"] == [(100, 125)]
    assert result["missed"] == 1
    assert result["mean_delay"] == 25

def test_empty_and_negative_tolerance():
    assert score_events([100], [], 20)["mean_delay"] is None

    with pytest.raises(ValueError):
        score_events([], [], -1)

def test_duplicate_alarm():
    result = score_events([100], [110, 111], 20)

    assert result["matched"] == 1
    assert result["false_alarms"] == 1

def test_no_changes():
    result = score_events([], [10, 20], 20)

    assert result["matched"] == 0
    assert result["false_alarms"] == 2
    assert result["missed"] == 0

def test_sequential_alarms():
    class StubDetector:
        def __init__(self):
            self.count = 0
            self.drift_detected = False
        
        def update(self, value):
            self.count += 1
            self.drift_detected = self.count in (3,5)

    assert collect_alarms(StubDetector(), range(6)) == [2,4]