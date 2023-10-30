"""Evaluation metrics.

All the metrics are updated one sample at a time. This way we can track performance of
predictive methods over time.

Note that all metrics have a `revert` method, enabling them to be wrapped in `utils.Rolling`.
This allows computirng rolling metrics:

```py
from river import metrics, utils

y_true = [True, False, True, True]
y_pred = [False, False, True, True]

metric = utils.Rolling(metrics.Accuracy(), window_size=3)

for yt, yp in zip(y_true, y_pred):
    print(metric.update(yt, yp))
```

```
Accuracy: 0.00%
Accuracy: 50.00%
Accuracy: 66.67%
Accuracy: 100.00%
```

"""
from __future__ import annotations

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
from .mape import MAPE
from .mcc import MCC
from .mse import MSE, RMSE, RMSLE
from .mutual_info import AdjustedMutualInfo, MutualInfo, NormalizedMutualInfo
from .precision import MacroPrecision, MicroPrecision, Precision, WeightedPrecision
from .r2 import R2
from .rand import AdjustedRand, Rand
from .recall import MacroRecall, MicroRecall, Recall, WeightedRecall
from .report import ClassificationReport
from .roc_auc import ROCAUC
from .rolling_roc_auc import RollingROCAUC
from .silhouette import Silhouette
from .smape import SMAPE
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
    "MAPE",
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
    "RollingROCAUC",
    "R2",
    "Precision",
    "Silhouette",
    "SMAPE",
    "VBeta",
    "WeightedF1",
    "WeightedFBeta",
    "WeightedPrecision",
    "WeightedRecall",
    "WeightedJaccard",
]
