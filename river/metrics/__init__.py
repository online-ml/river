"""Evaluation metrics.

All the metrics are updated one sample at a time. This way we can track performance of
predictive methods over time.

"""

from ._performance_evaluator import _ClassificationReport  # noqa: F401
from ._performance_evaluator import _MLClassificationReport  # noqa: F401
from ._performance_evaluator import _MTRegressionReport  # noqa: F401
from ._performance_evaluator import _RegressionReport  # noqa: F401
from ._performance_evaluator import _RollingClassificationReport  # noqa: F401
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
from .jaccard import Jaccard
from .kappa import CohenKappa, KappaM, KappaT
from .log_loss import LogLoss
from .mae import MAE
from .matthews_corrcoef import MatthewsCorrCoef
from .mcc import MCC
from .mse import MSE, RMSE, RMSLE
from .multioutput import RegressionMultiOutput
from .mutual_info import (
    AdjustedMutualInfo,
    ExpectedMutualInfo,
    MutualInfo,
    NormalizedMutualInfo,
)
from .pair_confusion import PairConfusionMatrix
from .precision import (
    ExamplePrecision,
    MacroPrecision,
    MicroPrecision,
    Precision,
    WeightedPrecision,
)
from .purity import Purity
from .r2 import R2
from .rand import AdjustedRand, Rand
from .recall import ExampleRecall, MacroRecall, MicroRecall, Recall, WeightedRecall
from .report import ClassificationReport
from .roc_auc import ROCAUC
from .rolling import Rolling
from .smape import SMAPE
from .time_rolling import TimeRolling
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
    "CohenKappa",
    "ConfusionMatrix",
    "CrossEntropy",
    "ExactMatch",
    "ExamplePrecision",
    "ExampleRecall",
    "ExampleF1",
    "ExampleFBeta",
    "ExpectedMutualInfo",
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
    "Rand",
    "Recall",
    "RegressionMetric",
    "RegressionMultiOutput",
    "RMSE",
    "RMSLE",
    "ROCAUC",
    "Rolling",
    "R2",
    "Precision",
    "SMAPE",
    "TimeRolling",
    "VBeta",
    "WeightedF1",
    "WeightedFBeta",
    "WeightedPrecision",
    "WeightedRecall",
    "WrapperMetric",
]
