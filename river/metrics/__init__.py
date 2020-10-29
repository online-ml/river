"""Evaluation metrics.

Note that the binary classification metrics expect the ground truths you provide them with to be
boolean values. In other words you need to pass a value of type `bool` to the `y_true` argument in
the `update` method of each binary metric. You will obtain incorrect results if instead you pass a
0 or 1 integer.

"""

from .accuracy import Accuracy
from .balanced_accuracy import BalancedAccuracy
from .base import Metric
from .confusion import ConfusionMatrix
from .cross_entropy import CrossEntropy
from .exact_match import ExactMatch
from .fbeta import F1
from .fbeta import FBeta
from .fbeta import MacroF1
from .fbeta import MacroFBeta
from .fbeta import MicroF1
from .fbeta import MicroFBeta
from .fbeta import MultiFBeta
from .fbeta import WeightedF1
from .fbeta import WeightedFBeta
from .fbeta import ExampleF1
from .fbeta import ExampleFBeta
from .geometric_mean import GeometricMean
from .hamming import Hamming
from .hamming import HammingLoss
from .jaccard import Jaccard
from .kappa import CohenKappa
from .kappa import KappaM
from .kappa import KappaT
from .log_loss import LogLoss
from .mae import MAE
from .mcc import MCC
from .mse import MSE
from .mse import RMSE
from .mse import RMSLE
from .multioutput import RegressionMultiOutput
from .precision import MacroPrecision
from .precision import MicroPrecision
from .precision import Precision
from .precision import WeightedPrecision
from .precision import ExamplePrecision
from .recall import MacroRecall
from .recall import MicroRecall
from .recall import Recall
from .recall import WeightedRecall
from .recall import ExampleRecall
from .report import ClassificationReport
from .roc_auc import ROCAUC
from .rolling import Rolling
from .smape import SMAPE
from .time_rolling import TimeRolling
from ._performance_evaluator import _ClassificationReport
from ._performance_evaluator import _RollingClassificationReport
from ._performance_evaluator import _MLClassificationReport
from ._performance_evaluator import _RollingMLClassificationReport
from ._performance_evaluator import _RegressionReport
from ._performance_evaluator import _RollingRegressionReport
from ._performance_evaluator import _MTRegressionReport
from ._performance_evaluator import _RollingMTRegressionReport

__all__ = [
    'Accuracy',
    'BalancedAccuracy',
    'ClassificationReport',
    'CohenKappa',
    'ConfusionMatrix',
    'CrossEntropy',
    'ExactMatch',
    'ExamplePrecision',
    'ExampleRecall',
    'ExampleF1',
    'ExampleFBeta',
    'F1',
    'FBeta',
    'GeometricMean',
    'Hamming',
    'HammingLoss',
    'Jaccard',
    'KappaM',
    'KappaT',
    'LogLoss',
    'MAE',
    'MacroF1',
    'MacroFBeta',
    'MacroPrecision',
    'MacroRecall',
    'MCC',
    'Metric',
    'MicroF1',
    'MicroFBeta',
    'MicroPrecision',
    'MicroRecall',
    'MultiFBeta',
    'MSE',
    'Precision',
    'Recall',
    'RegressionMultiOutput',
    'RMSE',
    'RMSLE',
    'ROCAUC',
    'Rolling',
    'SMAPE',
    'TimeRolling',
    'WeightedF1',
    'WeightedFBeta',
    'WeightedPrecision',
    'WeightedRecall'
]
