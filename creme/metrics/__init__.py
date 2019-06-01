"""
A set of metrics used in machine learning that can be computed in a streaming fashion, without any
loss in precision.
"""
from .accuracy import Accuracy
from .accuracy import RollingAccuracy
from .confusion import ConfusionMatrix
from .cross_entropy import CrossEntropy
from .cross_entropy import RollingCrossEntropy
from .f1_score import F1Score
from .f1_score import MacroF1Score
from .f1_score import MicroF1Score
from .f1_score import RollingF1Score
from .f1_score import RollingMacroF1Score
from .f1_score import RollingMicroF1Score
from .jaccard import Jaccard
from .log_loss import LogLoss
from .log_loss import RollingLogLoss
from .mae import MAE
from .mae import RollingMAE
from .mse import MSE
from .mse import RollingMSE
from .multioutput import RegressionMultiOutput
from .precision import MacroPrecision
from .precision import MicroPrecision
from .precision import Precision
from .precision import RollingMacroPrecision
from .precision import RollingMicroPrecision
from .precision import RollingPrecision
from .recall import MacroRecall
from .recall import MicroRecall
from .recall import Recall
from .recall import RollingMacroRecall
from .recall import RollingMicroRecall
from .recall import RollingRecall
from .rmse import RMSE
from .rmse import RollingRMSE
from .rmsle import RMSLE
from .rmsle import RollingRMSLE
from .smape import RollingSMAPE
from .smape import SMAPE


__all__ = [
    'Accuracy',
    'ConfusionMatrix',
    'CrossEntropy',
    'F1Score',
    'Jaccard',
    'LogLoss',
    'MacroF1Score',
    'MacroPrecision',
    'MacroRecall',
    'MicroF1Score',
    'MicroPrecision',
    'MicroRecall',
    'MAE',
    'MSE',
    'Precision',
    'Recall',
    'RegressionMultiOutput',
    'RMSE',
    'RMSLE',
    'RollingAccuracy',
    'RollingCrossEntropy',
    'RollingF1Score',
    'RollingLogLoss',
    'RollingMacroF1Score',
    'RollingMacroPrecision',
    'RollingMacroRecall',
    'RollingMAE',
    'RollingMicroF1Score',
    'RollingMicroPrecision',
    'RollingMicroRecall',
    'RollingMSE',
    'RollingPrecision',
    'RollingRecall',
    'RollingRMSE',
    'RollingRMSLE',
    'RollingSMAPE',
    'SMAPE'
]
