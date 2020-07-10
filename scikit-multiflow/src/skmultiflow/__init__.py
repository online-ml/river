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

from ._version import __version__
from .utils._show_versions import show_versions

__all__ = ['__version__', 'bayes', 'core', 'data', 'drift_detection', 'evaluation', 'lazy',
           'meta', 'metrics', 'neural_networks', 'trees', 'utils', 'visualization', 'show_versions']
