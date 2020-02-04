"""Feature extraction from a stream."""
from .agg import Agg
from .agg import TargetAgg
from .differ import Differ
from .vectorize import BagOfWords
from .vectorize import TFIDF


__all__ = [
    'Agg',
    'BagOfWords',
    'Differ',
    'TargetAgg',
    'TFIDF'
]
