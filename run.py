import itertools
import json
from tqdm import tqdm
from river import evaluate
from river import linear_model
from river import metrics
from river import preprocessing

models = {
    "Logisitic regression": preprocessing.StandardScaler() | linear_model.LogisticRegression()
}
track = evaluate.BinaryClassificationTrack()
results = []

for model_name, model in models.items():
    for run in track.run(model, n_checkpoints=1):
        res = next(run)
        res["Model"] = model_name
        res["Track"] = track.name
        for k, v in res.items():
            if isinstance(v, metrics.base.Metric):
                res[k] = v.get()
        res["Time"] = res["Time"].seconds
        results.append(res)

with open('results.json', 'w') as f:
    json.dump(results, f, sort_keys=True, indent=4)
