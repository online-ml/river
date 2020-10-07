"""Synthetic datasets.

Each synthetic dataset is an infinite generator.

"""
from .agrawal import Agrawal
from .anomaly_sine import AnomalySine
from .hyper_plane import Hyperplane
from .friedman import Friedman
from .sea import SEA


__all__ = [
    'Agrawal',
    'AnomalySine',
    'Friedman',
    'Hyperplane',
    'SEA'
]
