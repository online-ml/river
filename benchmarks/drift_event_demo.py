import numpy as np
import csv
import sys
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

def benchmark_cases():
    for name, (values, changes) in make_streams().items():
        yield name, values, {"synthetic": changes}

        for dataset in (
            drift.datasets.AirlinePassengers(),
            drift.datasets.Bitcoin(),
        ):
            refs = {
                f"ann_{k}": v
                for k,v in dataset.annotations.items()
            }

            for policy in ("union", "majority"):
                refs[policy] = sorted(dataset._annotations_aggregated(policy))

            yield type(dataset).__name__, [y for _, y in dataset], refs


def main():
    detectors = {
        "ADWIN": drift.ADWIN,
        "KSWIN": lambda: drift.KSWIN(seed=42),
        "PageHinkley": drift.PageHinkley,
    }

    writer = csv.writer(sys.stdout)
    writer.writerow([
        "stream", "reference", "detector", "policy",
        "tolerance", "matched", "missed", 
        "false_alarms", "mean_delay",
    ])

    for stream, values, references in benchmark_cases():
        alarms = {
            name: collect_alarms(factory(), values)
            for name, factory in detectors.items()
        }

        for reference, changes in references.items():
            for tolerance in (30, 120):
                rankings = {}

                for policy in ("oldest", "nearest"):
                    scores = {}

                    for name, observed in alarms.items():
                        result = score_events(
                            changes, observed, tolerance, policy
                        )
                        delay = result["mean_delay"]

                        scores[name] = (
                            result["missed"],
                            result["false_alarms"],
                            float("inf") if delay is None else delay,
                        )
                        writer.writerow([
                            stream, reference, name, policy, tolerance,
                            result["matched"], result["missed"],
                            result["false_alarms"],
                            "" if delay is None else delay,
                        ])
                    
                    rankings[policy] = sorted(
                        scores, key=lambda name:(scores[name], name)
                    )
                
                print(
                    f"{stream}/{reference} changes={len(changes)} "
                    f"tol={tolerance} "
                    f"oldest={rankings['oldest']} "
                    f"nearest={rankings['nearest']} "
                    f"changed={rankings['oldest'] != rankings['nearest']}",
                    file=sys.stderr
                )


if __name__ == "__main__":
    main()