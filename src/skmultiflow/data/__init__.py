"""
The :mod:`skmultiflow.data` module includes data stream methods including methods for batch-to-stream
conversion and generators.
"""

from .data_stream import DataStream
from .file_stream import FileStream
from .generators import *

__all__ = ["DataStream", "FileStream", "AGRAWALGenerator", "HyperplaneGenerator", "LEDGenerator", "LEDGeneratorDrift",
           "MIXEDGenerator", "MultilabelGenerator", "RandomRBFGenerator", "RandomRBFGeneratorDrift",
           "RandomTreeGenerator", "RegressionGenerator", "SEAGenerator", "SineGenerator", "STAGGERGenerator",
           "WaveformGenerator"]
