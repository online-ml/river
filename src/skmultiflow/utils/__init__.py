"""
The :mod:`skmultiflow.utils` module contains contains utility methods for `scikit-multiflow`.
"""

from .validation import check_random_state
from .validation import check_weights
from .utils import normalize_values_in_dict
from .utils import get_dimensions
from .data_structures import FastBuffer
from .data_structures import FastComplexBuffer

__all__ = ["check_random_state", "check_weights", "get_dimensions",
           "normalize_values_in_dict", "FastBuffer", "FastComplexBuffer"]
