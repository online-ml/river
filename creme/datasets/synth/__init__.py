"""Synthetic datasets.

Each synthetic dataset is an infinite generator.

"""
from .friedman import Friedman
from .sea import SEA


__all__ = [
    'Friedman',
    'SEA'
]
