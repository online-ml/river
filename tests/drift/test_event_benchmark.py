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

@pytest.mark.parametrize(
    "policy, expected, missed",
    [
        ("oldest", [(100, 125), (120, 145)], 0),
        ("nearest", [(120, 125)], 1)
    ]
)
def test_matching_policy_difference(policy, expected, missed):
    result = score_events([100, 120], [125, 145], 30, policy)

    assert result["matches"] == expected
    assert result["missed"] == missed

def test_annotation_disagreement():
    from river import drift

    dataset = drift.datasets.AirlinePassengers()

    assert dataset.annotations["6"] != dataset.annotations["9"]
    assert dataset._annotations_aggregated("union") == {
        296, 299, 302, 326, 382
    }
    assert dataset._annotations_aggregated("majority") == set()

def test_csv_schema(capsys):
    import csv
    import io

    from benchmarks.drift_event_demo import main

    main()
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))

    assert rows
    assert all(None not in row for row in rows)
    assert all(row["false_alarms"].isdigit() for row in rows)
    assert all(
        row["mean_delay"] == ""
        or float(row["mean_delay"]) >= 0
        for row in rows
    )

def test_invalid_matching_policy():
    with pytest.raises(ValueError):
        score_events([], [], policy="unsupported")