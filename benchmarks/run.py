import copy
import itertools
import multiprocessing
from datetime import timedelta
from typing import List

import pandas as pd

from config import MODELS, TRACKS, N_CHECKPOINTS
from river import metrics
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
from tqdm import tqdm

def run_dataset(model_str,no_dataset, no_track):
    model_name = model_str
    track = TRACKS[no_track]
    dataset = track.datasets[no_dataset]
    MODELS["Binary classification"].update(MODELS["Multiclass classification"])
    model = MODELS[track.name][model_name].clone()
    print(f'Processing {model_str} on {dataset.__class__.__name__}')

    results = []
    track = copy.deepcopy(track)
    time = 0.0
    for i in tqdm(track.run(model, copy.deepcopy(dataset), n_checkpoints=N_CHECKPOINTS), total=N_CHECKPOINTS):
        time += i['Time'] / timedelta (days=1)
        res = {
            "step": i["Step"],
            "track": track.name,
            "model": model_name,
            "dataset": dataset.__class__.__name__,
        }
        for k, v in i.items():
            if isinstance(v, metrics.base.Metric):
                res[k] = v.get()
        res["Memory"] = i['Memory'] / 1024 ** 2
        res["Time"] = time
        results.append(res)
    return results

def run_track(models: List[str], no_track: int, n_workers: int = 50):
    pool = multiprocessing.Pool(processes=n_workers)
    track = TRACKS[no_track]
    runs = list(itertools.product(models, range(len(track.datasets)), [no_track]))
    results = []

    for val in pool.starmap(run_dataset, runs):
        results.extend(val)
    pd.DataFrame(results).to_csv(f"{track.name}.csv", index=False)


if __name__ == '__main__':

    MODELS["Binary classification"].update(MODELS["Multiclass classification"])

    # Create details for each model
    for i, track in enumerate(TRACKS):
        run_track(models=MODELS[track.name].keys(), no_track=i, n_workers=70)