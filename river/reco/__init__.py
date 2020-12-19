"""Recommender systems."""
from .baseline import Baseline
from .biased_mf import BiasedMF
from .funk_mf import FunkMF
from .normal import RandomNormal


__all__ = ["Baseline", "BiasedMF", "FunkMF", "RandomNormal"]

try:
    from .surprise import SurpriseWrapper

    __all__ += ["SurpriseWrapper"]
except ImportError:
    pass
