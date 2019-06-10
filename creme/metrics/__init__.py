"""
A set of metrics used in machine learning that can be computed in a streaming fashion, without any
loss in precision.
"""
from .accuracy import Accuracy
from .accuracy import RollingAccuracy
from .confusion import ConfusionMatrix
from .confusion import RollingConfusionMatrix
from .cross_entropy import CrossEntropy
from .cross_entropy import RollingCrossEntropy
from .f1 import F1
from .f1 import MicroF1
from .f1 import MultiF1
from .f1 import RollingF1
from .f1 import RollingMicroF1
from .f1 import RollingMultiF1
from .fbeta import FBeta
from .fbeta import MicroFBeta
from .fbeta import MultiFBeta
from .fbeta import RollingFBeta
from .fbeta import RollingMicroFBeta
from .fbeta import RollingMultiFBeta
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
    'F1',
    'FBeta',
    'Jaccard',
    'LogLoss',
    'MacroPrecision',
    'MacroRecall',
    'MicroF1',
    'MicroFBeta'
    'MicroPrecision',
    'MicroRecall',
    'MultiF1',
    'MultiFBeta',
    'MAE',
    'MSE',
    'Precision',
    'Recall',
    'RegressionMultiOutput',
    'RMSE',
    'RMSLE',
    'RollingAccuracy',
    'RollingConfusionMatrix',
    'RollingCrossEntropy',
    'RollingF1',
    'RollingFBeta',
    'RollingLogLoss',
    'RollingMacroPrecision',
    'RollingMacroRecall',
    'RollingMAE',
    'RollingMicroF1',
    'RollingMicroFBeta',
    'RollingMicroPrecision',
    'RollingMicroRecall',
    'RollingMultiF1',
    'RollingMultiFBeta',
    'RollingMSE',
    'RollingPrecision',
    'RollingRecall',
    'RollingRMSE',
    'RollingRMSLE',
    'RollingSMAPE',
    'SMAPE'
]
