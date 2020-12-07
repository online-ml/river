"""Synthetic datasets.

Each synthetic dataset is a stream generator. The benefit of using a generator is that they do not
store the data and each data sample is generated on the fly. Except for a couple of methods,
the majority of these methods are infinite data generators.

"""
from .agrawal import Agrawal
from .anomaly_sine import AnomalySine
from .concept_drift_stream import ConceptDriftStream
from .friedman import Friedman
from .friedman import FriedmanDrift
from .hyper_plane import Hyperplane
from .led import LED
from .led import LEDDrift
from .logical import Logical
from .mixed import Mixed
from .mv import Mv
from .planes_2d import Planes2D
from .random_rbf import RandomRBF
from .random_rbf import RandomRBFDrift
from .random_tree import RandomTree
from .sea import SEA
from .sine import Sine
from .stagger import STAGGER
from .waveform import Waveform


__all__ = [
    "Agrawal",
    "AnomalySine",
    "ConceptDriftStream",
    "Friedman",
    "FriedmanDrift",
    "Hyperplane",
    "LED",
    "LEDDrift",
    "Logical",
    "Mixed",
    "Mv",
    "Planes2D",
    "RandomRBF",
    "RandomRBFDrift",
    "RandomTree",
    "SEA",
    "Sine",
    "STAGGER",
    "Waveform",
]
