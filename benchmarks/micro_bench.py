"""Microbenchmarks for River core modules.

Measures per-call latency of learn_one and predict_one for key models,
plus standalone benchmarks for metrics, stats, preprocessing, and stream.
"""

from __future__ import annotations

import gc
import json
import random
import sys
import time
import tracemalloc
from collections import defaultdict

import numpy as np

from river import (
    cluster,
    datasets,
    ensemble,
    evaluate,
    forest,
    linear_model,
    metrics,
    naive_bayes,
    neighbors,
    neural_net,
    optim,
    preprocessing,
    stats,
    stream,
    tree,
)

random.seed(42)
np.random.seed(42)

# ─── Helpers ───────────────────────────────────────────────────────────────

def make_numeric_sample(n_features=10):
    return {f"f{i}": random.gauss(0, 1) for i in range(n_features)}


def make_mixed_sample(n_numeric=8, n_categorical=2):
    d = {f"f{i}": random.gauss(0, 1) for i in range(n_numeric)}
    for i in range(n_categorical):
        d[f"c{i}"] = random.choice(["a", "b", "c", "d"])
    return d


def bench_model(name, model, make_x, make_y, n_warmup=200, n_iter=2000):
    """Benchmark learn_one + predict_one for a model."""
    # Warmup
    for _ in range(n_warmup):
        x, y = make_x(), make_y()
        model.learn_one(x, y)

    # Benchmark predict_one
    xs = [make_x() for _ in range(n_iter)]
    gc.disable()
    t0 = time.perf_counter()
    for x in xs:
        model.predict_one(x)
    predict_time = time.perf_counter() - t0
    gc.enable()

    # Benchmark learn_one
    ys = [make_y() for _ in range(n_iter)]
    gc.disable()
    t0 = time.perf_counter()
    for x, y in zip(xs, ys):
        model.learn_one(x, y)
    learn_time = time.perf_counter() - t0
    gc.enable()

    # Memory
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()
    fresh_model = model.clone()
    for _ in range(n_iter):
        fresh_model.learn_one(make_x(), make_y())
    snapshot2 = tracemalloc.take_snapshot()
    mem_diff = sum(stat.size_diff for stat in snapshot2.compare_to(snapshot1, 'lineno'))
    tracemalloc.stop()

    return {
        "model": name,
        "predict_one_us": predict_time / n_iter * 1e6,
        "learn_one_us": learn_time / n_iter * 1e6,
        "n_iter": n_iter,
        "mem_after_train_kb": mem_diff / 1024,
    }


def bench_classifier(name, model, make_x, n_classes=2, **kwargs):
    if n_classes == 2:
        make_y = lambda: random.choice([True, False])
    else:
        make_y = lambda: random.randint(0, n_classes - 1)
    return bench_model(name, model, make_x, make_y, **kwargs)


def bench_regressor(name, model, make_x, **kwargs):
    make_y = lambda: random.gauss(0, 1)
    return bench_model(name, model, make_x, make_y, **kwargs)


def bench_clusterer(name, model, make_x, n_warmup=200, n_iter=2000):
    for _ in range(n_warmup):
        model.learn_one(make_x())

    xs = [make_x() for _ in range(n_iter)]

    gc.disable()
    t0 = time.perf_counter()
    for x in xs:
        model.predict_one(x)
    predict_time = time.perf_counter() - t0
    gc.enable()

    gc.disable()
    t0 = time.perf_counter()
    for x in xs:
        model.learn_one(x)
    learn_time = time.perf_counter() - t0
    gc.enable()

    return {
        "model": name,
        "predict_one_us": predict_time / n_iter * 1e6,
        "learn_one_us": learn_time / n_iter * 1e6,
        "n_iter": n_iter,
    }


# ─── Module benchmarks ────────────────────────────────────────────────────

def bench_linear_models():
    print("\n=== LINEAR MODELS ===")
    results = []
    make_x = lambda: make_numeric_sample(20)

    models = {
        "LogisticRegression": linear_model.LogisticRegression(),
        "LogisticRegression(Adam)": linear_model.LogisticRegression(optimizer=optim.Adam()),
        "LinearRegression": linear_model.LinearRegression(),
        "LinearRegression(l1)": linear_model.LinearRegression(l1=1.0),
        "LinearRegression(l2)": linear_model.LinearRegression(l2=1.0),
        "PAClassifier": linear_model.PAClassifier(),
        "PARegressor": linear_model.PARegressor(),
        "Perceptron": linear_model.Perceptron(),
        "SoftmaxRegression": linear_model.SoftmaxRegression(),
        "BayesianLinearRegression": linear_model.BayesianLinearRegression(),
    }

    classifiers = {"LogisticRegression", "LogisticRegression(Adam)", "PAClassifier", "Perceptron", "SoftmaxRegression"}
    for name, model in models.items():
        if name in classifiers:
            r = bench_classifier(name, model, make_x, n_classes=3 if name == "SoftmaxRegression" else 2)
        else:
            r = bench_regressor(name, model, make_x)
        r["module"] = "linear_model"
        results.append(r)
        print(f"  {name}: learn={r['learn_one_us']:.1f}µs predict={r['predict_one_us']:.1f}µs")

    return results


def bench_tree_models():
    print("\n=== TREE MODELS ===")
    results = []
    make_x = lambda: make_numeric_sample(15)

    classifiers = {
        "HoeffdingTreeClassifier": tree.HoeffdingTreeClassifier(),
        "HoeffdingAdaptiveTreeClassifier": tree.HoeffdingAdaptiveTreeClassifier(seed=42),
        "ExtremelyFastDecisionTreeClassifier": tree.ExtremelyFastDecisionTreeClassifier(),
    }
    regressors = {
        "HoeffdingTreeRegressor": tree.HoeffdingTreeRegressor(),
        "HoeffdingAdaptiveTreeRegressor": tree.HoeffdingAdaptiveTreeRegressor(seed=42),
        "SGTRegressor": tree.SGTRegressor(),
    }

    for name, model in classifiers.items():
        r = bench_classifier(name, model, make_x, n_classes=5)
        r["module"] = "tree"
        results.append(r)
        print(f"  {name}: learn={r['learn_one_us']:.1f}µs predict={r['predict_one_us']:.1f}µs")

    for name, model in regressors.items():
        r = bench_regressor(name, model, make_x)
        r["module"] = "tree"
        results.append(r)
        print(f"  {name}: learn={r['learn_one_us']:.1f}µs predict={r['predict_one_us']:.1f}µs")

    return results


def bench_ensemble_models():
    print("\n=== ENSEMBLE MODELS ===")
    results = []
    make_x = lambda: make_numeric_sample(10)

    models_cls = {
        "BaggingClassifier(HT, n=10)": ensemble.BaggingClassifier(
            model=tree.HoeffdingTreeClassifier(), n_models=10, seed=42
        ),
        "ADWINBaggingClassifier(HT, n=10)": ensemble.ADWINBaggingClassifier(
            model=tree.HoeffdingTreeClassifier(), n_models=10, seed=42
        ),
        "AdaBoostClassifier(HT, n=10)": ensemble.AdaBoostClassifier(
            model=tree.HoeffdingTreeClassifier(), n_models=10, seed=42
        ),
        "LeveragingBaggingClassifier(HT, n=10)": ensemble.LeveragingBaggingClassifier(
            model=tree.HoeffdingTreeClassifier(), n_models=10, seed=42
        ),
        "SRPClassifier(n=10)": ensemble.SRPClassifier(n_models=10, seed=42),
        "ARFClassifier(n=10)": forest.ARFClassifier(n_models=10, seed=42),
    }
    models_reg = {
        "BaggingRegressor(HT, n=10)": ensemble.BaggingRegressor(
            model=tree.HoeffdingTreeRegressor(), n_models=10, seed=42
        ),
        "SRPRegressor(n=10)": ensemble.SRPRegressor(n_models=10, seed=42),
        "ARFRegressor(n=10)": forest.ARFRegressor(n_models=10, seed=42),
    }

    for name, model in models_cls.items():
        r = bench_classifier(name, model, make_x, n_classes=3, n_warmup=100, n_iter=500)
        r["module"] = "ensemble"
        results.append(r)
        print(f"  {name}: learn={r['learn_one_us']:.1f}µs predict={r['predict_one_us']:.1f}µs")

    for name, model in models_reg.items():
        r = bench_regressor(name, model, make_x, n_warmup=100, n_iter=500)
        r["module"] = "ensemble"
        results.append(r)
        print(f"  {name}: learn={r['learn_one_us']:.1f}µs predict={r['predict_one_us']:.1f}µs")

    return results


def bench_preprocessing():
    print("\n=== PREPROCESSING ===")
    results = []
    make_x_num = lambda: make_numeric_sample(20)
    make_x_mixed = lambda: make_mixed_sample(15, 5)
    n_iter = 5000

    transformers = {
        "StandardScaler": (preprocessing.StandardScaler(), make_x_num),
        "MinMaxScaler": (preprocessing.MinMaxScaler(), make_x_num),
        "MaxAbsScaler": (preprocessing.MaxAbsScaler(), make_x_num),
        "RobustScaler": (preprocessing.RobustScaler(), make_x_num),
        "Normalizer": (preprocessing.Normalizer(), make_x_num),
        "AdaptiveStandardScaler": (preprocessing.AdaptiveStandardScaler(), make_x_num),
        "OneHotEncoder": (preprocessing.OneHotEncoder(), make_x_mixed),
        "FeatureHasher": (preprocessing.FeatureHasher(n_features=50), make_x_mixed),
    }

    for name, (t, make_x) in transformers.items():
        # Warmup
        for _ in range(200):
            x = make_x()
            t.learn_one(x)

        xs = [make_x() for _ in range(n_iter)]

        gc.disable()
        t0 = time.perf_counter()
        for x in xs:
            t.transform_one(x)
        transform_time = time.perf_counter() - t0
        gc.enable()

        gc.disable()
        t0 = time.perf_counter()
        for x in xs:
            t.learn_one(x)
        learn_time = time.perf_counter() - t0
        gc.enable()

        r = {
            "model": name,
            "module": "preprocessing",
            "learn_one_us": learn_time / n_iter * 1e6,
            "predict_one_us": transform_time / n_iter * 1e6,
            "n_iter": n_iter,
        }
        results.append(r)
        print(f"  {name}: learn={r['learn_one_us']:.1f}µs transform={r['predict_one_us']:.1f}µs")

    return results


def bench_metrics():
    print("\n=== METRICS ===")
    results = []
    n_iter = 10000

    metric_instances = {
        "Accuracy": metrics.Accuracy(),
        "Precision": metrics.Precision(),
        "Recall": metrics.Recall(),
        "F1": metrics.F1(),
        "ROCAUC": metrics.ROCAUC(),
        "BalancedAccuracy": metrics.BalancedAccuracy(),
        "MAE": metrics.MAE(),
        "MSE": metrics.MSE(),
        "RMSE": metrics.RMSE(),
        "R2": metrics.R2(),
        "LogLoss": metrics.LogLoss(),
        "CrossEntropy": metrics.CrossEntropy(),
    }

    for name, m in metric_instances.items():
        is_regression = name in {"MAE", "MSE", "RMSE", "R2"}
        is_proba = name in {"ROCAUC", "LogLoss", "CrossEntropy"}

        if is_regression:
            yt = [random.gauss(0, 1) for _ in range(n_iter)]
            yp = [y + random.gauss(0, 0.1) for y in yt]
        elif is_proba:
            yt = [random.choice([True, False]) for _ in range(n_iter)]
            yp = [
                {True: random.random(), False: 1 - random.random()}
                for _ in range(n_iter)
            ]
            # Normalize
            yp = [{k: v / sum(d.values()) for k, v in d.items()} for d in yp]
        else:
            yt = [random.choice([True, False]) for _ in range(n_iter)]
            yp = [random.choice([True, False]) for _ in range(n_iter)]

        gc.disable()
        t0 = time.perf_counter()
        for y_true, y_pred in zip(yt, yp):
            m.update(y_true, y_pred)
        update_time = time.perf_counter() - t0
        gc.enable()

        gc.disable()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            m.get()
        get_time = time.perf_counter() - t0
        gc.enable()

        r = {
            "model": name,
            "module": "metrics",
            "learn_one_us": update_time / n_iter * 1e6,
            "predict_one_us": get_time / n_iter * 1e6,
            "n_iter": n_iter,
        }
        results.append(r)
        print(f"  {name}: update={r['learn_one_us']:.1f}µs get={r['predict_one_us']:.1f}µs")

    return results


def bench_stats():
    print("\n=== STATS ===")
    results = []
    n_iter = 20000

    stat_instances = {
        "Mean": stats.Mean(),
        "Var": stats.Var(),
        "Min": stats.Min(),
        "Max": stats.Max(),
        "Sum": stats.Sum(),
        "Count": stats.Count(),
        "Quantile(0.5)": stats.Quantile(0.5),
        "Cov": stats.Cov(),
        "PearsonCorr": stats.PearsonCorr(),
        "Kurtosis": stats.Kurtosis(),
        "Skew": stats.Skew(),
        "EWMean(0.1)": stats.EWMean(fading_factor=0.1),
        "EWVar(0.1)": stats.EWVar(fading_factor=0.1),
        "RollingMax(100)_2": stats.RollingMax(window_size=100),
        "RollingMax(100)": stats.RollingMax(window_size=100),
        "RollingMin(100)": stats.RollingMin(window_size=100),
        "RollingQuantile(100, 0.5)": stats.RollingQuantile(window_size=100, q=0.5),
        "Entropy": stats.Entropy(),
        "IQR": stats.IQR(),
    }

    bivariate = {"Cov", "PearsonCorr"}

    for name, s in stat_instances.items():
        values = [random.gauss(0, 1) for _ in range(n_iter)]

        gc.disable()
        t0 = time.perf_counter()
        if name in bivariate:
            for v in values:
                s.update(v, v + random.gauss(0, 0.1))
        else:
            for v in values:
                s.update(v)
        update_time = time.perf_counter() - t0
        gc.enable()

        gc.disable()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            s.get()
        get_time = time.perf_counter() - t0
        gc.enable()

        r = {
            "model": name,
            "module": "stats",
            "learn_one_us": update_time / n_iter * 1e6,
            "predict_one_us": get_time / n_iter * 1e6,
            "n_iter": n_iter,
        }
        results.append(r)
        print(f"  {name}: update={r['learn_one_us']:.1f}µs get={r['predict_one_us']:.1f}µs")

    return results


def bench_cluster():
    print("\n=== CLUSTER ===")
    results = []
    make_x = lambda: make_numeric_sample(10)

    models = {
        "KMeans(k=5)": cluster.KMeans(n_clusters=5, seed=42),
        "STREAMKMeans(k=5)": cluster.STREAMKMeans(
            chunk_size=50, n_clusters=5, seed=42
        ),
        "DenStream": cluster.DenStream(),
        "DBSTREAM": cluster.DBSTREAM(),
        "CluStream": cluster.CluStream(seed=42),
    }

    for name, model in models.items():
        r = bench_clusterer(name, model, make_x, n_warmup=200, n_iter=1000)
        r["module"] = "cluster"
        results.append(r)
        print(f"  {name}: learn={r['learn_one_us']:.1f}µs predict={r['predict_one_us']:.1f}µs")

    return results


def bench_stream():
    print("\n=== STREAM ===")
    results = []

    # Bench iter_array
    n_samples = 5000
    X = np.random.randn(n_samples, 20)
    y = np.random.randn(n_samples)

    gc.disable()
    t0 = time.perf_counter()
    for _ in stream.iter_array(X, y):
        pass
    iter_array_time = time.perf_counter() - t0
    gc.enable()

    r = {
        "model": "iter_array(5000x20)",
        "module": "stream",
        "learn_one_us": iter_array_time / n_samples * 1e6,
        "predict_one_us": 0,
        "n_iter": n_samples,
    }
    results.append(r)
    print(f"  iter_array: {r['learn_one_us']:.1f}µs/sample")

    # Bench iter_csv - create a temp CSV
    import tempfile, csv, os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        writer = csv.writer(f)
        writer.writerow([f"f{i}" for i in range(20)] + ["target"])
        for i in range(n_samples):
            writer.writerow([random.gauss(0, 1) for _ in range(20)] + [random.gauss(0, 1)])
        tmpfile = f.name

    gc.disable()
    t0 = time.perf_counter()
    count = 0
    for _ in stream.iter_csv(tmpfile, target="target"):
        count += 1
    iter_csv_time = time.perf_counter() - t0
    gc.enable()
    os.unlink(tmpfile)

    r = {
        "model": "iter_csv(5000x20)",
        "module": "stream",
        "learn_one_us": iter_csv_time / count * 1e6,
        "predict_one_us": 0,
        "n_iter": count,
    }
    results.append(r)
    print(f"  iter_csv: {r['learn_one_us']:.1f}µs/sample")

    # Bench dataset iteration
    for ds_name, ds in [("Phishing", datasets.Phishing()), ("Elec2", datasets.Elec2())]:
        gc.disable()
        t0 = time.perf_counter()
        count = 0
        for _ in ds:
            count += 1
        ds_time = time.perf_counter() - t0
        gc.enable()

        r = {
            "model": f"dataset:{ds_name}",
            "module": "stream",
            "learn_one_us": ds_time / count * 1e6,
            "predict_one_us": 0,
            "n_iter": count,
        }
        results.append(r)
        print(f"  {ds_name} iteration: {r['learn_one_us']:.1f}µs/sample ({count} samples)")

    return results


def bench_evaluate():
    print("\n=== EVALUATE ===")
    results = []

    # Benchmark progressive_val_score on ChickWeights (regression)
    model = preprocessing.StandardScaler() | linear_model.LinearRegression()
    dataset = datasets.ChickWeights()

    gc.disable()
    t0 = time.perf_counter()
    metric = evaluate.progressive_val_score(dataset, model, metrics.MAE(), print_every=0)
    eval_time = time.perf_counter() - t0
    gc.enable()

    n_samples = 578
    r = {
        "model": "progressive_val_score(LR+Scaler, ChickWeights)",
        "module": "evaluate",
        "learn_one_us": eval_time / n_samples * 1e6,
        "predict_one_us": 0,
        "n_iter": n_samples,
        "total_time_s": eval_time,
    }
    results.append(r)
    print(f"  progressive_val_score (LR, ChickWeights): {eval_time:.2f}s total, {r['learn_one_us']:.1f}µs/sample")

    # With a tree model
    model2 = tree.HoeffdingTreeRegressor()
    dataset2 = datasets.ChickWeights()

    gc.disable()
    t0 = time.perf_counter()
    metric2 = evaluate.progressive_val_score(dataset2, model2, metrics.MAE(), print_every=0)
    eval_time2 = time.perf_counter() - t0
    gc.enable()

    r2 = {
        "model": "progressive_val_score(HT, ChickWeights)",
        "module": "evaluate",
        "learn_one_us": eval_time2 / n_samples * 1e6,
        "predict_one_us": 0,
        "n_iter": n_samples,
        "total_time_s": eval_time2,
    }
    results.append(r2)
    print(f"  progressive_val_score (HT, ChickWeights): {eval_time2:.2f}s total, {r2['learn_one_us']:.1f}µs/sample")

    return results


def bench_optimize():
    print("\n=== OPTIMIZERS ===")
    results = []
    n_iter = 10000

    optimizers = {
        "SGD(0.01)": optim.SGD(0.01),
        "Adam(0.01)": optim.Adam(0.01),
        "AdaGrad(0.01)": optim.AdaGrad(0.01),
        "RMSProp(0.01)": optim.RMSProp(0.01),
        "Momentum(0.01)": optim.Momentum(0.01),
        "AdaBound(0.01)": optim.AdaBound(0.01),
        "AdaDelta": optim.AdaDelta(),
        "FTRLProximal": optim.FTRLProximal(),
    }

    n_features = 20

    for name, opt in optimizers.items():
        # Simulate gradient steps
        weights = {f"f{i}": 0.0 for i in range(n_features)}
        gradients = [{f"f{i}": random.gauss(0, 1) for i in range(n_features)} for _ in range(n_iter)]

        gc.disable()
        t0 = time.perf_counter()
        for g in gradients:
            weights = opt.step(weights, g)
        step_time = time.perf_counter() - t0
        gc.enable()

        r = {
            "model": name,
            "module": "optim",
            "learn_one_us": step_time / n_iter * 1e6,
            "predict_one_us": 0,
            "n_iter": n_iter,
        }
        results.append(r)
        print(f"  {name}: step={r['learn_one_us']:.1f}µs")

    return results


def bench_pipelines():
    """Benchmark pipeline overhead by comparing raw model vs pipeline."""
    print("\n=== PIPELINE OVERHEAD ===")
    results = []
    make_x = lambda: make_numeric_sample(20)
    make_y = lambda: random.choice([True, False])
    n_iter = 2000

    # Raw model
    raw = linear_model.LogisticRegression()
    pipe = preprocessing.StandardScaler() | linear_model.LogisticRegression()

    for name, model in [("LogisticRegression (raw)", raw), ("Scaler | LogisticRegression (pipe)", pipe)]:
        for _ in range(200):
            model.learn_one(make_x(), make_y())

        xs = [make_x() for _ in range(n_iter)]
        ys = [make_y() for _ in range(n_iter)]

        gc.disable()
        t0 = time.perf_counter()
        for x, y in zip(xs, ys):
            model.learn_one(x, y)
        learn_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        for x in xs:
            model.predict_one(x)
        predict_time = time.perf_counter() - t0
        gc.enable()

        r = {
            "model": name,
            "module": "compose",
            "learn_one_us": learn_time / n_iter * 1e6,
            "predict_one_us": predict_time / n_iter * 1e6,
            "n_iter": n_iter,
        }
        results.append(r)
        print(f"  {name}: learn={r['learn_one_us']:.1f}µs predict={r['predict_one_us']:.1f}µs")

    return results


def bench_feature_scaling():
    """Benchmark impact of feature count on key operations."""
    print("\n=== SCALING WITH FEATURES ===")
    results = []

    for n_features in [5, 20, 50, 100, 200]:
        make_x = lambda nf=n_features: make_numeric_sample(nf)
        model = linear_model.LogisticRegression()
        r = bench_classifier(f"LogReg(n_feat={n_features})", model, make_x)
        r["module"] = "scaling"
        r["n_features"] = n_features
        results.append(r)
        print(f"  n_features={n_features}: learn={r['learn_one_us']:.1f}µs predict={r['predict_one_us']:.1f}µs")

    for n_features in [5, 20, 50, 100, 200]:
        make_x = lambda nf=n_features: make_numeric_sample(nf)
        model = tree.HoeffdingTreeClassifier()
        r = bench_classifier(f"HoeffdingTree(n_feat={n_features})", model, make_x, n_classes=3)
        r["module"] = "scaling"
        r["n_features"] = n_features
        results.append(r)
        print(f"  HT n_features={n_features}: learn={r['learn_one_us']:.1f}µs predict={r['predict_one_us']:.1f}µs")

    return results


# ─── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_results = []

    all_results.extend(bench_linear_models())
    all_results.extend(bench_tree_models())
    all_results.extend(bench_ensemble_models())
    all_results.extend(bench_preprocessing())
    all_results.extend(bench_metrics())
    all_results.extend(bench_stats())
    all_results.extend(bench_cluster())
    all_results.extend(bench_stream())
    all_results.extend(bench_evaluate())
    all_results.extend(bench_optimize())
    all_results.extend(bench_pipelines())
    all_results.extend(bench_feature_scaling())

    # Save results
    with open("micro_bench_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Module':<18} {'Model':<42} {'learn_one(µs)':<15} {'predict_one(µs)':<15}")
    print("=" * 90)
    for r in sorted(all_results, key=lambda x: x.get("learn_one_us", 0), reverse=True):
        print(f"{r['module']:<18} {r['model']:<42} {r['learn_one_us']:>12.1f}   {r['predict_one_us']:>12.1f}")
    print("=" * 90)
