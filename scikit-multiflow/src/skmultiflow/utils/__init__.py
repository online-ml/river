"""
The :mod:`skmultiflow.utils` module contains contains utility methods for `scikit-multiflow`.
"""

from .validation import check_random_state
from .validation import check_weights
from .utils import normalize_values_in_dict
from .utils import get_dimensions
from .utils import is_scalar_nan
from .utils import calculate_object_size
from .utils import get_max_value_key
from .utils import add_dict_values
from .utils import add_delay_to_timestamps
from .data_structures import FastBuffer
from .data_structures import FastComplexBuffer
from .data_structures import SlidingWindow
from ._show_versions import show_versions

__all__ = ["check_random_state", "check_weights", "calculate_object_size", "get_dimensions",
           "is_scalar_nan", "normalize_values_in_dict", "FastBuffer", "FastComplexBuffer",
           "show_versions", "get_max_value_key", "add_dict_values", "SlidingWindow",
           "add_delay_to_timestamps"]
