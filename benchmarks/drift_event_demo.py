import numpy as np
from river import drift
from river.drift.event_benchmark import collect_alarms, score_events

def make_streams():
    rng = np.random.RandomState(12345)

    abrupt = np.concatenate((
        rng.randint(2, size=1000),
        rng.randint(8, size=1000),
    ))

    rng = np.random.RandomState(12345)

    recurring = np.concatenate((
        rng.normal(0.0, 0.1, 500) > 0,
        rng.normal(0.25, 0.1, 500) > 0,
        rng.normal(0.0, 0.1, 500) > 0,
        rng.normal(0.25, 0.1, 500) > 0,
    )).astype(int)

    return {
        "stationery": (np.ones(1000), []),
        "abrupt": (abrupt, [1000]),
        "recurring": (recurring, [500, 1000, 1500]),
    }


def main():
    detectors = {
        "ADWIN": drift.ADWIN,
        "KSWIN": lambda: drift.KSWIN(seed=42),
        "PageHinkley": drift.PageHinkley,
    }

    print(
        "stream, detector, tolerance,"
        "matched, missed, false_alarms, mean_delay"
    )

    for stream_name, (values, changes) in make_streams().items():
        for detector_name, make_detector in detectors.items():
            alarms = collect_alarms(make_detector(), values)

            for tolerance in (30, 120):
                result = score_events(changes, alarms, tolerance)
                delay = result["mean_delay"]

                print(
                    f"{stream_name},{detector_name}, {tolerance},"
                    f"{result['matched']},{result['missed']},"
                    f"{result['false_alarms'],}"
                    f"{'' if delay is None else delay}"
                )


if __name__ == "__main__":
    main()