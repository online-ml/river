"""Utilities for handling streaming datasets."""
from .cache import Cache
from .array import iter_array
from .array import iter_pandas
from .array import iter_sklearn_dataset
from .csv import iter_csv
from .libsvm import iter_libsvm
from .qa import simulate_qa
from .shuffle import shuffle
from .vaex import iter_vaex

__all__ = [
    'Cache',
    'iter_array',
    'iter_csv',
    'iter_libsvm',
    'iter_pandas',
    'iter_sklearn_dataset',
    'iter_sql',
    'iter_vaex',
    'simulate_qa',
    'shuffle'
]

try:
    from .sql import iter_sql

    __all__ += ['iter_sql']
except ImportError:
    pass
