"""
creme is a library for incremental learning. Incremental learning is a machine learning regime
where the observations are made available one by one. It is also known as online learning,
iterative learning, or sequential learning. This is in contrast to batch learning where all the
data is processed at once. Incremental learning is desirable when the data is too big to fit in
memory, or simply when it isn't available all at once. creme's API is heavily inspired from that of
scikit-learn, enough so that users who are familiar with scikit-learn should feel right at home.
"""
from .__version__ import __version__

from . import anomaly
from . import base
from . import cluster
from . import compat
from . import compose
from . import datasets
from . import dummy
from . import decomposition
from . import ensemble
from . import facto
from . import feature_extraction
from . import feature_selection
from . import impute
from . import linear_model
from . import meta
from . import metrics
from . import model_selection
from . import multiclass
from . import multioutput
from . import naive_bayes
from . import neighbors
from . import optim
from . import preprocessing
from . import proba
from . import reco
from . import sampling
from . import stats
from . import stream
from . import time_series
from . import tree
from . import utils

__all__ = [
    'anomaly',
    'base',
    'cluster',
    'compat',
    'compose',
    'datasets',
    'dummy',
    'decomposition',
    'ensemble',
    'facto',
    'feature_extraction',
    'feature_selection',
    'impute',
    'linear_model',
    'meta',
    'metrics',
    'model_selection',
    'multiclass',
    'multioutput',
    'naive_bayes',
    'neighbors',
    'optim',
    'preprocessing',
    'proba',
    'reco',
    'sampling',
    'stats',
    'stream',
    'time_series',
    'tree',
    'utils'
]

__pdoc__ = {
    'anomaly': False,
    'cluster': False,
    'compat': False,
    'compose': False,
    'datasets': False,
    'dummy': False,
    'decomposition': False,
    'ensemble': False,
    'facto': False,
    'feature_extraction': False,
    'feature_selection': False,
    'impute': False,
    'linear_model': False,
    'meta': False,
    'multiclass': False,
    'multioutput': False,
    'naive_bayes': False,
    'optim': False,
    'preprocessing': False,
    'proba': False,
    'reco': False,
    'sampling': False,
    'time_series': False,
    'tree': False,
    'utils': False,

    'conftest': False
}
