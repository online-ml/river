"""Evaluation metrics.

All the metrics are updated one sample at a time. This way we can track performance of
predictive methods over time.

"""

from . import cluster
from ._performance_evaluator import _ClassificationReport  # noqa: F401
from ._performance_evaluator import _ClusteringReport  # noqa: F401
from ._performance_evaluator import _MLClassificationReport  # noqa: F401
from ._performance_evaluator import _MTRegressionReport  # noqa: F401
from ._performance_evaluator import _RegressionReport  # noqa: F401
from ._performance_evaluator import _RollingClassificationReport  # noqa: F401
from ._performance_evaluator import _RollingClusteringReport  # noqa: F401
from ._performance_evaluator import _RollingMLClassificationReport  # noqa: F401
from ._performance_evaluator import _RollingMTRegressionReport  # noqa: F401
from ._performance_evaluator import _RollingRegressionReport  # noqa: F401
from .accuracy import Accuracy
from .balanced_accuracy import BalancedAccuracy
from .base import (
    BinaryMetric,
    ClassificationMetric,
    Metric,
    Metrics,
    MultiClassMetric,
    MultiOutputClassificationMetric,
    MultiOutputRegressionMetric,
    RegressionMetric,
    WrapperMetric,
)
from .confusion import ConfusionMatrix, MultiLabelConfusionMatrix
from .cross_entropy import CrossEntropy
from .exact_match import ExactMatch
from .expected_mutual_info import expected_mutual_info
from .fbeta import (
    F1,
    ExampleF1,
    ExampleFBeta,
    FBeta,
    MacroF1,
    MacroFBeta,
    MicroF1,
    MicroFBeta,
    MultiFBeta,
    WeightedF1,
    WeightedFBeta,
)
from .fowlkes_mallows import FowlkesMallows
from .geometric_mean import GeometricMean
from .hamming import Hamming, HammingLoss
from .jaccard import Jaccard, SorensenDice
from .kappa import CohenKappa, KappaM, KappaT
from .log_loss import LogLoss
from .mae import MAE
from .matthews_corrcoef import MatthewsCorrCoef
from .mcc import MCC
from .mse import MSE, RMSE, RMSLE
from .multioutput import RegressionMultiOutput
from .mutual_info import AdjustedMutualInfo, MutualInfo, NormalizedMutualInfo
from .pair_confusion import PairConfusionMatrix
from .precision import (
    ExamplePrecision,
    MacroPrecision,
    MicroPrecision,
    Precision,
    WeightedPrecision,
)
from .prevalence_threshold import PrevalenceThreshold
from .purity import Purity
from .q0 import Q0, Q2
from .r2 import R2
from .rand import AdjustedRand, Rand
from .recall import ExampleRecall, MacroRecall, MicroRecall, Recall, WeightedRecall
from .report import ClassificationReport
from .roc_auc import ROCAUC
from .rolling import Rolling
from .smape import SMAPE
from .time_rolling import TimeRolling
from .variation_info import VariationInfo
from .vbeta import Completeness, Homogeneity, VBeta

__all__ = [
    "Accuracy",
    "AdjustedMutualInfo",
    "AdjustedRand",
    "BalancedAccuracy",
    "BinaryMetric",
    "Completeness",
    "ClassificationMetric",
    "ClassificationReport",
    "ClusteringReport",
    "cluster",
    "CohenKappa",
    "ConfusionMatrix",
    "CrossEntropy",
    "ExactMatch",
    "ExamplePrecision",
    "ExampleRecall",
    "ExampleF1",
    "ExampleFBeta",
    "expected_mutual_info",
    "F1",
    "FBeta",
    "FowlkesMallows",
    "GeometricMean",
    "Hamming",
    "HammingLoss",
    "Homogeneity",
    "Jaccard",
    "KappaM",
    "KappaT",
    "LogLoss",
    "MAE",
    "MacroF1",
    "MacroFBeta",
    "MacroPrecision",
    "MacroRecall",
    "MatthewsCorrCoef",
    "MCC",
    "Metric",
    "Metrics",
    "MicroF1",
    "MicroFBeta",
    "MicroPrecision",
    "MicroRecall",
    "MultiClassMetric",
    "MultiFBeta",
    "MultiLabelConfusionMatrix",
    "MultiOutputClassificationMetric",
    "MultiOutputRegressionMetric",
    "MSE",
    "MutualInfo",
    "NormalizedMutualInfo",
    "PairConfusionMatrix",
    "Precision",
    "Purity",
    "Q0",
    "Q2",
    "Rand",
    "Recall",
    "RegressionMetric",
    "RegressionMultiOutput",
    "RMSE",
    "RMSLE",
    "ROCAUC",
    "Rolling",
    "RollingClusteringReport",
    "R2",
    "Precision",
    "PrevalenceThreshold",
    "SMAPE",
    "SorensenDice",
    "TimeRolling",
    "VariationInfo",
    "VBeta",
    "WeightedF1",
    "WeightedFBeta",
    "WeightedPrecision",
    "WeightedRecall",
    "WrapperMetric",
]
