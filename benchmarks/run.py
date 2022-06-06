import datetime as dt
import itertools
import json
from tqdm import tqdm

from river import drift
from river import ensemble
from river import evaluate
from river import linear_model
from river import metrics
from river import neighbors
from river import neural_net as nn
from river import optim
from river import preprocessing
from river import rules
from river import tree
from dominate.tags import *


def run_track(models, track):
    print(track.name)
    results = []
    for model_name, model in models.items():
        print(f"\t{model_name}")
        for dataset in track:
            res = next(track.run(model, dataset, n_checkpoints=1))
            res["Dataset"] = dataset.__class__.__name__
            res["Model"] = model_name
            for k, v in res.items():
                if isinstance(v, metrics.base.Metric):
                    res[k] = v.get()
            res["Time"] = res["Time"] / dt.timedelta(milliseconds=1)
            results.append(res)

    return results


scalers = {
    "MaxAbs": preprocessing.MaxAbsScaler(),
    "MinMax": preprocessing.MinMaxScaler(),
    "Standard": preprocessing.StandardScaler(),
    "Robust": preprocessing.RobustScaler(),
}

# Binary Classification
bin_class_models = {
    "Logistic regression": preprocessing.StandardScaler() | linear_model.LogisticRegression()
}

# Single-target Regression
regressors = {
    "Linear Regression": linear_model.LinearRegression(),
    "Linear Regression with l1 regularization": linear_model.LinearRegression(l1=1.0),
    "Linear Regression with l2 regularization": linear_model.LinearRegression(l2=1.0),
    "Passive-Aggressive Regressor, mode 1":  linear_model.PARegressor(mode=1),
    "Passive-Aggressive Regressor, mode 2": linear_model.PARegressor(mode=2),
    "k-Nearest Neighbors": neighbors.KNNRegressor(window_size=100),
    "Hoeffding Tree": tree.HoeffdingAdaptiveTreeRegressor(),
    "Hoeffding Adaptive Tree": tree.HoeffdingAdaptiveTreeRegressor(seed=42),
    "Adaptive Random Forest": ensemble.AdaptiveRandomForestRegressor(seed=42),
    "Adaptive Model Rules": rules.AMRules(drift_detector=drift.ADWIN()),
    "Multi-layer Perceptron": nn.MLPRegressor(
        hidden_dims=(5,),
        activations=(
            nn.activations.ReLU,
            nn.activations.ReLU,
            nn.activations.Identity
        ),
        optimizer=optim.SGD(1e-3),
        seed=42
    )
}


def reg_combinations():
    combinations = {}
    for scaler_n, scaler in scalers.items():
        for reg_n, reg in regressors.items():
            combinations[f"{scaler_n} | {reg_n}"] = scaler.clone() | reg.clone()

    return combinations


benchmarks = {}
bin_class_track = evaluate.BinaryClassificationTrack()
benchmarks[bin_class_track.name] = run_track(models=bin_class_models, track=bin_class_track)

reg_track = evaluate.RegressionTrack()
benchmarks[reg_track.name] = run_track(models=reg_combinations(), track=reg_track)


with open('results.json', 'w') as f:
    json.dump(benchmarks, f, sort_keys=True, indent=4)
