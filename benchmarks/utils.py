import river

import collections
import time

from river import base
import pandas as pd
import tqdm


def format_ns(d):

    units = collections.OrderedDict({"ns": 1})
    units["μs"] = 1000 * units["ns"]
    units["ms"] = 1000 * units["μs"]
    units["s"] = 1000 * units["ms"]
    units["m"] = 60 * units["s"]
    units["h"] = 60 * units["m"]
    units["d"] = 24 * units["h"]

    parts = []

    for unit in reversed(units):
        amount = units[unit]
        quotient, d = divmod(d, amount)
        if quotient > 0:
            parts.append(f"{quotient}{unit}")
        elif d == 0:
            break

    return ", ".join(parts)


def benchmark(get_X_y, n, get_pp, models, get_metric):

    Result = collections.namedtuple("Result", "lib model score learn_time pred_time")
    results = []

    for lib, name, model in tqdm.tqdm(models, position=0):

        pp = get_pp()
        metric = get_metric()
        learn_time = 0
        pred_time = 0

        # Determine if predict_one or predict_proba_one should be used in case of a classifier
        pred_func = model.predict_one
        if isinstance(model, base.Classifier) and not metric.requires_labels:
            pred_func = model.predict_proba_one

        for x, y in tqdm.tqdm(get_X_y(), total=n, position=0):

            x = pp.learn_one(x, y).transform_one(x)

            # Predict
            tic = time.perf_counter_ns()
            y_pred = pred_func(x)
            pred_time += time.perf_counter_ns() - tic

            # Score
            metric.update(y_true=y, y_pred=y_pred)

            # Fit
            tic = time.perf_counter_ns()
            model.learn_one(x, y)
            learn_time += time.perf_counter_ns() - tic

        results.append(Result(lib, name, metric.get(), learn_time, pred_time))

    results = pd.DataFrame(
        {
            "Library": [r.lib for r in results],
            "Model": [r.model for r in results],
            metric.__class__.__name__: [r.score for r in results],
            "Fit time": [format_ns(r.learn_time) for r in results],
            "Average fit time": [format_ns(round(r.learn_time / n)) for r in results],
            "Predict time": [format_ns(r.pred_time) for r in results],
            "Average predict time": [format_ns(round(r.pred_time / n)) for r in results],
        }
    )

    return results
