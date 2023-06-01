# from river.changepoints.methods.base import ChangePointDetector TODO: Change path for integration into river
from methods.base import ChangePointDetector
# import river.metrics.changepoints.base TODO: Change path for integration into river
import metrics.changepoints.base
# import river.datasets.changepoints.base TODO: Change path for integration into river
import datasets.base

import csv


def load_dataset(dataset: datasets.base.ChangePointDataset):
    """
    Load a changepoint dataset.
    :param dataset: The dataset to load.
    :return: A tuple containing the data and the annotations.
    """
    annotations = dataset.annotations
    return dataset, annotations


def run_method(method: ChangePointDetector, data: datasets.base.ChangePointDataset):
    """
    Run a changepoint detection method on a dataset.
    :param method: The method to run.
    :param data: The dataset to run the method on.
    :return: A list of change points.
    """
    changepoints = []
    n_obs = 0
    for t, (T, x) in enumerate(data, start=1):
        method.update(x, t)
        n_obs += 1
        if method.change_point_detected:
            changepoints.append(t)
    return changepoints, n_obs


def evaluate_method(
        annotations: list,
        changepoints: list,
        metric: metrics.changepoints.base.ChangePointMetric,
        n_obs: int):
    """
    Evaluate a method on a dataset.
    :param changepoints: A list of change points.
    :param annotations: A list of annotations.
    :param metric: The metric to use for evaluation.
    :param n_obs: The number of observations in the dataset.
    :return: A dictionary containing the results.
    """
    return metric(annotations, changepoints, n_obs=n_obs)


def benchmark(
        method,
        dataset,
        metric,
        to_csv=None,
        include_csv_header=True):
    if not isinstance(dataset, datasets.base.ChangePointDataset):
        results = []
        for ds in dataset:
            results.append(benchmark(method, ds, metric, to_csv=None))
            method._reset()
        if to_csv:
            with open(to_csv, "a") as f:
                writer = csv.writer(f)
                if include_csv_header:
                    writer.writerow(["Method", "Dataset", "Metric", "Result"])
                for i, ds in enumerate(dataset):
                    if isinstance(metric, metrics.changepoints.base.ChangePointMetrics):
                        for j, m in enumerate(metric):
                            if results[i] is not None:
                                writer.writerow([method, ds, m, results[i][j]])
                            else:
                                writer.writerow([method, ds, m, ""])
                    else:
                        if results[i] is not None:
                            writer.writerow([method, ds, metric, results[i]])
                        else:
                            writer.writerow([method, ds, metric, ""])
        return results

    data, annotations = load_dataset(dataset)
    if data.n_features > 1 and not method.is_multivariate():
        print(
            f"Method {method} cannot handle multivariate input sequences. Skipping dataset.")
        if to_csv:
            with open(to_csv, "a") as f:
                writer = csv.writer(f)
                if include_csv_header:
                    writer.writerow(["Method", "Dataset", "Metric", "Result"])
                if isinstance(metric, metrics.changepoints.base.ChangePointMetrics):
                    for i, m in enumerate(metric):
                        writer.writerow([method, dataset, m, ""])
                else:
                    writer.writerow([method, dataset, metric, ""])
        return None
    changepoints, n_obs = run_method(method, data)
    print("changepoints", changepoints)
    results = evaluate_method(annotations, changepoints, metric, n_obs=n_obs)
    if to_csv:
        with open(to_csv, "a") as f:
            writer = csv.writer(f)
            if include_csv_header:
                writer.writerow(["Method", "Dataset", "Metric", "Result"])
            if isinstance(metric, metrics.changepoints.base.ChangePointMetrics):
                for i, m in enumerate(metric):
                    writer.writerow([method, dataset, m, results[i]])
            else:
                writer.writerow([method, dataset, metric, results])
    return results
