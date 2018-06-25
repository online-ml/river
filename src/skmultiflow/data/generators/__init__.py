"""
The :mod:`skmultiflow.data.generators` module includes data stream generators.
"""

from .agrawal_generator import AGRAWALGenerator
from .hyper_plane_generator import HyperplaneGenerator
from .led_generator import LEDGenerator
from .led_generator_drift import LEDGeneratorDrift
from .mixed_generator import MIXEDGenerator
from .multilabel_generator import MultilabelGenerator
from .random_rbf_generator import RandomRBFGenerator
from .random_rbf_generator_drift import RandomRBFGeneratorDrift
from .random_tree_generator import RandomTreeGenerator
from .regression_generator import RegressionGenerator
from .sea_generator import SEAGenerator
from .sine_generator import SineGenerator
from .stagger_generator import STAGGERGenerator
from .waveform_generator import WaveformGenerator

__all__ = ["AGRAWALGenerator", "HyperplaneGenerator", "LEDGenerator", "LEDGeneratorDrift", "MIXEDGenerator",
           "MultilabelGenerator", "RandomRBFGenerator", "RandomRBFGeneratorDrift", "RandomTreeGenerator",
           "RegressionGenerator", "SEAGenerator", "SineGenerator", "STAGGERGenerator", "WaveformGenerator"]
