"""
river is a library for incremental learning. Incremental learning is a machine learning regime
where the observations are made available one by one. It is also known as online learning,
iterative learning, or sequential learning. This is in contrast to batch learning where all the
data is processed at once. Incremental learning is desirable when the data is too big to fit in
memory, or simply when it isn't available all at once. river's API is heavily inspired from that of
scikit-learn, enough so that users who are familiar with scikit-learn should feel right at home.
"""
from . import (
    anomaly,
    base,
    cluster,
    compat,
    compose,
    datasets,
    drift,
    dummy,
    ensemble,
    evaluate,
    selection,
    facto,
    feature_extraction,
    feature_selection,
    imblearn,
    linear_model,
    meta,
    metrics,
    multiclass,
    multioutput,
    naive_bayes,
    neighbors,
    neural_net,
    optim,
    preprocessing,
    proba,
    reco,
    rules,
    stats,
    stream,
    time_series,
    tree,
    utils,
)
from .__version__ import __version__  # noqa: F401
from .datasets import synth

__all__ = [
    "anomaly",
    "base",
    "cluster",
    "compat",
    "compose",
    "datasets",
    "dummy",
    "drift",
    "ensemble",
    "evaluate",
    "selection",
    "facto",
    "feature_extraction",
    "feature_selection",
    "imblearn",
    "linear_model",
    "meta",
    "metrics",
    "multiclass",
    "multioutput",
    "naive_bayes",
    "neighbors",
    "neural_net",
    "optim",
    "preprocessing",
    "proba",
    "reco",
    "rules",
    "stats",
    "stream",
    "synth",
    "time_series",
    "tree",
    "utils",
]
