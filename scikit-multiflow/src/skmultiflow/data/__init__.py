"""
The :mod:`skmultiflow.data` module contains data stream methods including methods for
batch-to-stream conversion and generators.
"""

from .data_stream import DataStream
from .temporal_data_stream import TemporalDataStream
from .file_stream import FileStream
from .agrawal_generator import AGRAWALGenerator
from .concept_drift_stream import ConceptDriftStream
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
from .synth import make_logical
from .waveform_generator import WaveformGenerator
from .time_manager import TimeManager
from .anomaly_sine_generator import AnomalySineGenerator

__all__ = ["DataStream", "TemporalDataStream", "FileStream", "AGRAWALGenerator",
           "ConceptDriftStream", "HyperplaneGenerator", "LEDGenerator", "LEDGeneratorDrift",
           "MIXEDGenerator", "MultilabelGenerator", "RandomRBFGenerator",
           "RandomRBFGeneratorDrift", "RandomTreeGenerator", "RegressionGenerator", "SEAGenerator",
           "SineGenerator", "STAGGERGenerator", "make_logical", "WaveformGenerator", "TimeManager",
           "AnomalySineGenerator"]
