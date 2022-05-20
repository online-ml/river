"""Evaluation metrics.

All the metrics are updated one sample at a time. This way we can track performance of
predictive methods over time.

"""
from . import base, multioutput
from .accuracy import Accuracy
from .balanced_accuracy import BalancedAccuracy
from .confusion import ConfusionMatrix
from .cross_entropy import CrossEntropy
from .fbeta import (
    F1,
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
from .jaccard import Jaccard, MacroJaccard, MicroJaccard, WeightedJaccard
from .kappa import CohenKappa
from .log_loss import LogLoss
from .mae import MAE
from .mcc import MCC
from .mse import MSE, RMSE, RMSLE
from .mutual_info import AdjustedMutualInfo, MutualInfo, NormalizedMutualInfo
from .precision import MacroPrecision, MicroPrecision, Precision, WeightedPrecision
from .r2 import R2
from .rand import AdjustedRand, Rand
from .recall import MacroRecall, MicroRecall, Recall, WeightedRecall
from .report import ClassificationReport
from .roc_auc import ROCAUC
from .rolling import Rolling
from .silhouette import Silhouette
from .smape import SMAPE
from .time_rolling import TimeRolling
from .vbeta import Completeness, Homogeneity, VBeta

__all__ = [
    "Accuracy",
    "AdjustedMutualInfo",
    "AdjustedRand",
    "BalancedAccuracy",
    "base",
    "Completeness",
    "ClassificationReport",
    "ClusteringReport",
    "CohenKappa",
    "ConfusionMatrix",
    "CrossEntropy",
    "Jaccard",
    "MacroJaccard",
    "MicroJaccard",
    "F1",
    "FBeta",
    "GeometricMean",
    "Homogeneity",
    "LogLoss",
    "MAE",
    "MacroF1",
    "MacroFBeta",
    "MacroPrecision",
    "MacroRecall",
    "MCC",
    "MicroF1",
    "MicroFBeta",
    "MicroPrecision",
    "MicroRecall",
    "MultiFBeta",
    "multioutput",
    "MSE",
    "MutualInfo",
    "NormalizedMutualInfo",
    "Precision",
    "Rand",
    "Recall",
    "RMSE",
    "FowlkesMallows",
    "RMSLE",
    "ROCAUC",
    "Rolling",
    "R2",
    "Precision",
    "Silhouette",
    "SMAPE",
    "TimeRolling",
    "VBeta",
    "WeightedF1",
    "WeightedFBeta",
    "WeightedPrecision",
    "WeightedRecall",
    "WeightedJaccard",
]
