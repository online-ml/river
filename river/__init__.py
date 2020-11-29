"""
river is a library for incremental learning. Incremental learning is a machine learning regime
where the observations are made available one by one. It is also known as online learning,
iterative learning, or sequential learning. This is in contrast to batch learning where all the
data is processed at once. Incremental learning is desirable when the data is too big to fit in
memory, or simply when it isn't available all at once. river's API is heavily inspired from that of
scikit-learn, enough so that users who are familiar with scikit-learn should feel right at home.
"""
from .__version__ import __version__  # noqa: F401

from . import anomaly
from . import base
from . import cluster
from . import compat
from . import compose
from . import datasets
from . import dummy
from . import drift
from . import ensemble
from . import evaluate
from . import expert
from . import facto
from . import feature_extraction
from . import feature_selection
from . import imblearn
from . import linear_model
from . import meta
from . import metrics
from . import multiclass
from . import multioutput
from . import naive_bayes
from . import neighbors
from . import optim
from . import preprocessing
from . import proba
from . import reco
from . import stats
from . import stream
from . import time_series
from . import tree
from . import utils
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
    "expert",
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
    "optim",
    "preprocessing",
    "proba",
    "reco",
    "stats",
    "stream",
    "synth",
    "time_series",
    "tree",
    "utils",
]
