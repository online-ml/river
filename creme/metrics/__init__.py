"""Evaluation metrics.

Note that the binary classification metrics expect the ground truths you provide them with to be
boolean values. In other words you need to pass a value of type `bool` to the `y_true` argument in
the `update` method of each binary metric. You will obtain incorrect results if instead you pass a
0 or 1 integer.

"""

from .accuracy import Accuracy
from .base import Metric
from .confusion import ConfusionMatrix
from .cross_entropy import CrossEntropy
from .fbeta import F1
from .fbeta import FBeta
from .fbeta import MacroF1
from .fbeta import MacroFBeta
from .fbeta import MicroF1
from .fbeta import MicroFBeta
from .fbeta import MultiFBeta
from .fbeta import WeightedF1
from .fbeta import WeightedFBeta
from .jaccard import Jaccard
from .log_loss import LogLoss
from .mae import MAE
from .mcc import MCC
from .mse import MSE
from .multioutput import RegressionMultiOutput
from .precision import MacroPrecision
from .precision import MicroPrecision
from .precision import Precision
from .precision import WeightedPrecision
from .recall import MacroRecall
from .recall import MicroRecall
from .recall import Recall
from .recall import WeightedRecall
from .report import ClassificationReport
from .rmse import RMSE
from .rmsle import RMSLE
from .roc_auc import ROCAUC
from .rolling import Rolling
from .smape import SMAPE
from .time_rolling import TimeRolling


__all__ = [
    'Accuracy',
    'ClassificationReport',
    'ConfusionMatrix',
    'CrossEntropy',
    'F1',
    'FBeta',
    'Jaccard',
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
    'MSE',
    'MultiFBeta',
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
