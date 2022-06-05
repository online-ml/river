import itertools
import json
from tqdm import tqdm
from river import evaluate
from river import linear_model
from river import metrics
from river import preprocessing
from dominate.tags import *


benchmarks = {}

models = {
    "Logistic regression": preprocessing.StandardScaler() | linear_model.LogisticRegression()
}
track = evaluate.BinaryClassificationTrack()
results = []

for model_name, model in models.items():
    for dataset in track:
        res = next(track.run(model, dataset, n_checkpoints=1))
        res["Dataset"] = dataset.__class__.__name__
        res["Model"] = model_name
        for k, v in res.items():
            if isinstance(v, metrics.base.Metric):
                res[k] = v.get()
        res["Time"] = res["Time"].microseconds
        results.append(res)

benchmarks[track.name] = results

with open('results.json', 'w') as f:
    json.dump(benchmarks, f, sort_keys=True, indent=4)
