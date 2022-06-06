import datetime
import json
import shelve
import sys

from river import (
    drift,
    dummy,
    ensemble,
    evaluate,
    linear_model,
    metrics,
    naive_bayes,
    neighbors,
)
from river import neural_net as nn
from river import optim, preprocessing, rules, stats, tree


def run_track(models, track, benchmarks):
    print(track.name)
    if track.name in benchmarks:
        completed = set((cr["Dataset"], cr["Model"]) for cr in dict(benchmarks[track.name]))
    else:
        completed = set()

    for model_name, model in models.items():
        print(f"\t{model_name}")
        for dataset in track:
            data_name = dataset.__class__.__name__
            if (data_name, model_name) in completed:
                print(f"\t\t[skipped] {data_name}")
                continue
            # Get cached data from the shelf
            results = benchmarks[track.name]
            res = next(track.run(model, dataset, n_checkpoints=1))
            res["Dataset"] = data_name
            res["Model"] = model_name
            for k, v in res.items():
                if isinstance(v, metrics.base.Metric):
                    res[k] = v.get()
            res["Time"] = res["Time"] / datetime.timedelta(milliseconds=1)
            results.append(res)

            # Writes updated version to the shelf
            benchmarks[track.name] = results
            print(f"\t\t[done] {data_name}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "force":
        benchmarks = shelve.open("results.db", protocol="n")
    else:
        benchmarks = shelve.open("results.db", protocol="c")

    # Binary Classification
    bin_class_models = {
        # Binary only
        "Logistic regression": preprocessing.StandardScaler()
        | linear_model.LogisticRegression(),
        "ALMA": preprocessing.StandardScaler() | linear_model.ALMAClassifier(),
        "Stochastic Gradient Tree": tree.SGTClassifier(),
    }

    # Multiclass Classification
    multi_class_models = {
        "Naive Bayes": naive_bayes.GaussianNB(),
        "Hoeffding Tree": tree.HoeffdingTreeClassifier(),
        "Hoeffding Adaptive Tree": tree.HoeffdingAdaptiveTreeClassifier(seed=42),
        "Extremely Fast Decision Tree": tree.ExtremelyFastDecisionTreeClassifier(),
        "Adaptive Random Forest": ensemble.AdaptiveRandomForestClassifier(seed=42),
        "Streaming Random Patches": ensemble.SRPClassifier(),
        "k-Nearest Neighbors": preprocessing.StandardScaler()
        | neighbors.KNNClassifier(window_size=100),
        "ADWIN Bagging": ensemble.ADWINBaggingClassifier(
            tree.HoeffdingTreeClassifier(), seed=42
        ),
        "AdaBoost": ensemble.AdaBoostClassifier(
            tree.HoeffdingTreeClassifier(), seed=42
        ),
        "Bagging": ensemble.BaggingClassifier(tree.HoeffdingTreeClassifier(), seed=42),
        "Leveraging Bagging": ensemble.LeveragingBaggingClassifier(
            tree.HoeffdingTreeClassifier(), seed=42
        ),
        "Stacking": ensemble.StackingClassifier(
            [
                preprocessing.StandardScaler() | linear_model.SoftmaxRegression(),
                naive_bayes.GaussianNB(),
                tree.HoeffdingTreeClassifier(),
                preprocessing.StandardScaler()
                | neighbors.KNNClassifier(window_size=100),
            ],
            meta_classifier=ensemble.AdaptiveRandomForestClassifier(seed=42),
        ),
        # "Voting": ensemble.VotingClassifier(
        #     [
        #         preprocessing.StandardScaler() | linear_model.SoftmaxRegression(),
        #         naive_bayes.GaussianNB(),
        #         tree.HoeffdingTreeClassifier(),
        #         preprocessing.StandardScaler()
        #         | neighbors.KNNClassifier(window_size=100),
        #     ]
        # ),
        # Baseline
        "[baseline] Last Class": dummy.NoChangeClassifier(),
    }

    # Single-target Regression
    regressors = {
        "Linear Regression": preprocessing.StandardScaler()
        | linear_model.LinearRegression(),
        "Linear Regression with l1 regularization": preprocessing.StandardScaler()
        | linear_model.LinearRegression(l1=1.0),
        "Linear Regression with l2 regularization": preprocessing.StandardScaler()
        | linear_model.LinearRegression(l2=1.0),
        "Passive-Aggressive Regressor, mode 1": preprocessing.StandardScaler()
        | linear_model.PARegressor(mode=1),
        "Passive-Aggressive Regressor, mode 2": preprocessing.StandardScaler()
        | linear_model.PARegressor(mode=2),
        "k-Nearest Neighbors": preprocessing.StandardScaler()
        | neighbors.KNNRegressor(window_size=100),
        "Hoeffding Tree": preprocessing.StandardScaler()
        | tree.HoeffdingAdaptiveTreeRegressor(),
        "Hoeffding Adaptive Tree": preprocessing.StandardScaler()
        | tree.HoeffdingAdaptiveTreeRegressor(seed=42),
        "Stochastic Gradient Tree": tree.SGTRegressor(),
        "Adaptive Random Forest": preprocessing.StandardScaler()
        | ensemble.AdaptiveRandomForestRegressor(seed=42),
        "Adaptive Model Rules": preprocessing.StandardScaler()
        | rules.AMRules(drift_detector=drift.ADWIN()),
        "Streaming Random Patches": preprocessing.StandardScaler()
        | ensemble.SRPRegressor(seed=42),
        "Exponentially Weighted Average": preprocessing.StandardScaler()
        | ensemble.EWARegressor(
            models=[
                linear_model.LinearRegression(),
                tree.HoeffdingAdaptiveTreeRegressor(),
                neighbors.KNNRegressor(window_size=100),
                rules.AMRules(),
            ],
        ),
        "Multi-layer Perceptron": preprocessing.StandardScaler()
        | nn.MLPRegressor(
            hidden_dims=(5,),
            activations=(
                nn.activations.ReLU,
                nn.activations.ReLU,
                nn.activations.Identity,
            ),
            optimizer=optim.SGD(1e-3),
            seed=42,
        ),
        # Baseline
        "[baseline] Mean predictor": dummy.StatisticRegressor(stats.Mean()),
    }

    # Also include multiclass models
    bin_class_models.update(multi_class_models)
    bin_class_track = evaluate.BinaryClassificationTrack()
    run_track(
        models=bin_class_models, track=bin_class_track, benchmarks=benchmarks
    )

    multi_class_track = evaluate.MultiClassClassificationTrack()
    run_track(
        models=multi_class_models, track=multi_class_track, benchmarks=benchmarks
    )

    reg_track = evaluate.RegressionTrack()
    run_track(
        models=regressors, track=reg_track, benchmarks=benchmarks
    )

    # Create json dump with all the results
    with open("results.json", "w") as f:
        json.dump(dict(benchmarks), f, sort_keys=True, indent=4)

    # Close the shelf
    benchmarks.close()

    # Save info about the compared models and datasets
    benchmark_info = {}
    bin_c = {
        "Dataset": {},
        "Model": {}
    }
    for dataset in bin_class_track:
        bin_c["Dataset"][dataset.__class__.name] = repr(dataset)
    for model_n, model in bin_class_models.items():
        bin_c["Model"][model_n] = repr(model)
    benchmark_info["Binary Classification"] = bin_c

    multi_c = {
        "Dataset": {},
        "Model": {}
    }
    for dataset in multi_class_track:
        multi_c["Dataset"][dataset.__class__.name] = repr(dataset)
    for model_n, model in multi_class_models.items():
        multi_c[model_n] = repr(model)
    benchmark_info["Multiclass Classification"] = multi_c

    reg = {
        "Dataset": {},
        "Model": {}
    }
    for dataset in reg_track:
        reg["Dataset"][dataset.__class__.name] = repr(dataset)
    for model_n, model in regressors.items():
        reg[model_n] = repr(model)
    benchmark_info["Single-target regression"] = reg

    with open("model-info.json", "w") as f:
        json.dump(benchmark_info, f, sort_keys=True, indent=4)
