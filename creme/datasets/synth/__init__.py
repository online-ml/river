"""Synthetic datasets.

Each synthetic dataset is an infinite generator.

"""
from .agrawal import Agrawal
from .anomaly_sine import AnomalySine
from .friedman import Friedman
from .hyper_plane import Hyperplane
from .led import LED
from .led import LEDDrift
from .sea import SEA


__all__ = [
    'Agrawal',
    'AnomalySine',
    'Friedman',
    'Hyperplane',
    'LED',
    'LEDDrift',
    'SEA'
]
