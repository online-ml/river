"""
creme is a library for incremental learning. Incremental learning is a machine learning regime
where the observations are made available one by one. It is also known as online learning,
iterative learning, or sequential learning. This is in contrast to batch learning where all the
data is processed at once. Incremental learning is desirable when the data is too big to fit in
memory, or simply when it isn't available all at once. creme's API is heavily inspired from that of
scikit-learn, enough so that users who are familiar with scikit-learn should feel right at home.
"""
from . import cluster
from . import compat
from . import compose
from . import datasets
from . import ensemble
from . import feature_extraction
from . import feature_selection
from . import impute
from . import linear_model
from . import metrics
from . import model_selection
from . import multiclass
from . import naive_bayes
from . import optim
from . import preprocessing
from . import reco
from . import stats
from . import stream

from .__version__ import __version__

__all__ = [
    'cluster',
    'compat',
    'compose',
    'datasets',
    'ensemble',
    'feature_extraction',
    'feature_selection',
    'impute',
    'linear_model',
    'metrics',
    'model_selection',
    'multiclass',
    'naive_bayes',
    'optim',
    'preprocessing',
    'reco',
    'stats',
    'stream'
]
