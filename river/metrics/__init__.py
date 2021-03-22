"""Evaluation metrics.

All the metrics are updated one sample at a time. This way we can track performance of
predictive methods over time.

"""

from . import cluster
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
from .geometric_mean import GeometricMean
from .hamming import Hamming, HammingLoss
from .jaccard import Jaccard
from .kappa import CohenKappa, KappaM, KappaT
from .log_loss import LogLoss
from .mae import MAE
from .mcc import MCC
from .mse import MSE, RMSE, RMSLE
from .multioutput import RegressionMultiOutput
from .precision import (
    ExamplePrecision,
    MacroPrecision,
    MicroPrecision,
    Precision,
    WeightedPrecision,
)
from .r2 import R2
from .recall import ExampleRecall, MacroRecall, MicroRecall, Recall, WeightedRecall
from .report import ClassificationReport
from .roc_auc import ROCAUC
from .rolling import Rolling
from .smape import SMAPE
from .time_rolling import TimeRolling

__all__ = [
    "Accuracy",
    "BalancedAccuracy",
    "BinaryMetric",
    "ClassificationMetric",
    "ClassificationReport",
    "cluster",
    "CohenKappa",
    "ConfusionMatrix",
    "CrossEntropy",
    "ExactMatch",
    "ExamplePrecision",
    "ExampleRecall",
    "ExampleF1",
    "ExampleFBeta",
    "F1",
    "FBeta",
    "GeometricMean",
    "Hamming",
    "HammingLoss",
    "Jaccard",
    "KappaM",
    "KappaT",
    "LogLoss",
    "MAE",
    "MacroF1",
    "MacroFBeta",
    "MacroPrecision",
    "MacroRecall",
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
    "Precision",
    "Recall",
    "RegressionMetric",
    "RegressionMultiOutput",
    "RMSE",
    "RMSLE",
    "ROCAUC",
    "Rolling",
    "R2",
    "SMAPE",
    "TimeRolling",
    "WeightedF1",
    "WeightedFBeta",
    "WeightedPrecision",
    "WeightedRecall",
    "WrapperMetric",
]
