import itertools
import multiprocessing
import os

from benchmarks.config import MODELS, TRACKS

os.environ['KMP_DUPLICATE_LIB_OK']='True'


import datetime
import json
import shelve
import sys

from river import metrics

def run_track(models, track, benchmarks, n_workers=50):
    print(track.name)
    # Get cached data from the shelf
    if track.name in benchmarks:
        results = benchmarks[track.name]
    else:
        results = []

    paramlist = list(itertools.product(models.items(), track))
    pool = multiprocessing.Pool()

    # Distribute the parameter sets evenly across the cores
    res = pool.map(run_dataset, paramlist)


    for model_dict in models.items():
        for dataset in track:
            results.append(res)# Writes updated version to the shelf
            benchmarks[track.name] = results


def run_dataset(model_dict, dataset):
    model_name = model_dict[0]
    model = model_dict[1]
    print(f"\t{model_name}")  #
    dataset_name = dataset.__class__.__name__


    res = next(track.run(model, dataset, n_checkpoints=1))
    res["Dataset"] = dataset_name
    res["Model"] = model_name
    for k, v in res.items():
        if isinstance(v, metrics.base.Metric):
            res[k] = v.get()
    res["Time"] = res["Time"] / datetime.timedelta(milliseconds=1)
    res.pop("Step")
    print(f"\t\t[done] {dataset_name}")
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "force":
        benchmarks = shelve.open("results", flag="n")
    else:
        benchmarks = shelve.open("results", flag="c")

    # Every multiclass model can also handle binary classification
    MODELS["Binary classification"].update(MODELS["Multiclass classification"])
    details = {}

    for track in TRACKS:
        run_track(models=MODELS[track.name], track=track, benchmarks=benchmarks, n_workers=20)
        details[track.name] = {"Dataset": {}, "Model": {}}
        for dataset in track.datasets:
            details[track.name]["Dataset"][dataset.__class__.__name__] = repr(dataset)
        for model_name, model in MODELS[track.name].items():
            details[track.name]["Model"][model_name] = repr(model)

    log = {}
    for track in TRACKS:
        if track.name in benchmarks:
            log[track.name] = benchmarks[track.name]

    with open("results.json", "w") as f:
        json.dump(log, f, sort_keys=True, indent=4)

    # Close the shelf
    benchmarks.close()

    with open("details.json", "w") as f:
        json.dump(details, f, sort_keys=True, indent=4)
