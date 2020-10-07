"""Synthetic datasets.

Each synthetic dataset is an infinite generator.

"""
from .agrawal import Agrawal
from .anomaly_sine import AnomalySine
from .friedman import Friedman
from .sea import SEA


__all__ = [
    'Agrawal',
    'AnomalySine',
    'Friedman',
    'SEA'
]
