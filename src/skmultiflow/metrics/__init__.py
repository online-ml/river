"""
The :mod:`skmultiflow.metrics` module includes performance metrics.
"""

from ._classification_performance_evaluator import ClassificationPerformanceEvaluator
from ._classification_performance_evaluator import WindowClassificationPerformanceEvaluator
from ._classification_performance_evaluator import MultiLabelClassificationPerformanceEvaluator
from ._classification_performance_evaluator import WindowMultiLabelClassificationPerformanceEvaluator
from .measure_collection import RegressionMeasurements
from .measure_collection import MultiTargetRegressionMeasurements
from .measure_collection import WindowRegressionMeasurements
from .measure_collection import WindowMultiTargetRegressionMeasurements
from .measure_collection import RunningTimeMeasurements
from .measure_collection import ConfusionMatrix
from .measure_collection import MOLConfusionMatrix
from .measure_collection import hamming_score
from .measure_collection import exact_match
from .measure_collection import j_index
# To be removed in v0.7
from .measure_collection import ClassificationMeasurements
from .measure_collection import MultiTargetClassificationMeasurements
from .measure_collection import WindowClassificationMeasurements
from .measure_collection import WindowMultiTargetClassificationMeasurements


__all__ = ["ClassificationPerformanceEvaluator", "WindowClassificationPerformanceEvaluator",
           "MultiLabelClassificationPerformanceEvaluator", "WindowMultiLabelClassificationPerformanceEvaluator",
           "ClassificationMeasurements", "RegressionMeasurements",
           "MultiTargetClassificationMeasurements", "MultiTargetRegressionMeasurements",
           "WindowClassificationMeasurements", "WindowRegressionMeasurements",
           "WindowMultiTargetClassificationMeasurements",
           "WindowMultiTargetRegressionMeasurements",
           "RunningTimeMeasurements",
           "ConfusionMatrix", "MOLConfusionMatrix", "hamming_score",
           "exact_match", "j_index"]
