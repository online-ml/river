"""
This module implements the Attribute Observers (AO) (or tree splitters) that are used by the
iDTs. AOs are a core aspect of the iDT construction, and might represent one of the major
bottlenecks when building the trees. The correct choice and setup of a splitter might result
in significant differences in the running time and memory usage of the iDTs.

Splitters for classification and regression trees can be differentiated by using the property
`is_target_class` (`True` for splitters designed to classification tasks). An error will be raised
if one tries to use a classification splitter in a regression tree and vice-versa.

"""

from .base_splitter import Splitter
from .ebst_splitter import EBSTSplitter
from .exhaustive_splitter import ExhaustiveSplitter
from .gaussian_splitter import GaussianSplitter
from .histogram_splitter import HistogramSplitter
from .qo_splitter import QOSplitter
from .tebst_splitter import TEBSTSplitter

__all__ = [
    "Splitter",
    "ExhaustiveSplitter",
    "GaussianSplitter",
    "HistogramSplitter",
    "EBSTSplitter",
    "QOSplitter",
    "TEBSTSplitter",
]
