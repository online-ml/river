"""Utility classes and functions."""
from .estimator_checks import check_estimator
from .estimator_checks import guess_model
from .histogram import Histogram
from .math import chain_dot
from .math import clamp
from .math import dot
from .math import minkowski_distance
from .math import norm
from .math import prod
from .math import sigmoid
from .math import softmax
from .pretty import pretty_format_class
from .sdft import SDFT
from .skyline import Skyline
from .window import Window
from .window import SortedWindow


__all__ = [
    'chain_dot',
    'check_estimator',
    'clamp',
    'dot',
    'Histogram',
    'minkowski_distance',
    'norm',
    'pretty_format_class',
    'prod',
    'SDFT',
    'sigmoid',
    'Skyline',
    'softmax',
    'SortedWindow',
    'Window'
]
