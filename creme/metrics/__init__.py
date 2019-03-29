"""
A set of metrics used in machine learning that can be computed in a streaming
fashion.
"""
from .accuracy import Accuracy
from .confusion import ConfusionMatrix
from .f1_score import F1Score
from .f1_score import MacroF1Score
from .f1_score import MicroF1Score
from .log_loss import LogLoss
from .mae import MAE
from .mse import MSE
from .precision import MacroPrecision
from .precision import MicroPrecision
from .precision import Precision
from .recall import MacroRecall
from .recall import MicroRecall
from .recall import Recall
from .rmse import RMSE
from .rmsle import RMSLE
from .smape import SMAPE


__all__ = [
    'Accuracy',
    'ConfusionMatrix',
    'F1Score',
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
    'RMSE',
    'RMSLE',
    'SMAPE'
]
