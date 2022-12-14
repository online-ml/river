import itertools
import multiprocessing

import pandas as pd

from config import MODELS, TRACKS
import json
from river import metrics, evaluate
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def run_dataset(model,dataset, track):
    model_name = model[0]
    model = model[1]
    results = []
    for i in track.run(model, dataset, n_checkpoints=50):
        res = {
            "step": i["Step"],
            "track": track.name,
            "model": model_name,
            "dataset": dataset.__class__.__name__,
            "Memory": i['Memory'],
            "Time": str(i['Time']),
        }
        for k, v in i.items():
            if isinstance(v, metrics.base.Metric):
                res[k] = v.get()
        results.append(res)
    return results

def run_track(models: dict, track: evaluate.Track, n_workers: int = 5):
    #pool = multiprocessing.Pool(processes=n_workers)
    #runs = itertools.product(models.items(), track.datasets, [track])
    #results = pool.starmap(run_dataset, runs)
    results = []
    for model in models.items():
        for dataset in track.datasets:
            results.extend(run_dataset(model,dataset,track))

    pd.DataFrame(results).to_csv(f"{track.name}.csv", index=False)
    #json.dump(results, open(f"{track.name}.json", "w+"), indent=4)




if __name__ == '__main__':

    MODELS["Binary classification"].update(MODELS["Multiclass classification"])

    # Create details for each model
    for track in TRACKS:
        run_track(models=MODELS[track.name], track=track, n_workers=5)