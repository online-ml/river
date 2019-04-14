""" A machine learning framework for multi-output/multi-label and stream data.
"""

from . import bayes
from . import core
from . import data
from . import drift_detection
from . import evaluation
from . import lazy
from . import meta
from . import metrics
from . import neural_networks
from . import trees
from . import utils
from . import visualization

__all__ = ['bayes', 'core', 'data', 'drift_detection', 'evaluation', 'lazy',
           'meta', 'metrics', 'neural_networks', 'trees', 'utils', 'visualization']
