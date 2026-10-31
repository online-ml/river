# Rolling computations

You might wonder which classes in River can be wrapped with a [`utils.Rolling`](<https://riverml.xyz/dev/api/utils/Rolling>). This can be answered with a bit of metaprogramming.

```python
import importlib
import inspect
from river.utils.rolling import Rollable

for submodule in importlib.import_module("river.api").__all__:
    for _, obj in inspect.getmembers(
        importlib.import_module(f"river.{submodule}"), lambda x: isinstance(x, Rollable)
    ):
        print(f'{submodule}.{obj.__name__}')
```

```text
covariance.EmpiricalCovariance
metrics.Accuracy
metrics.AdjustedMutualInfo
metrics.AdjustedRand
metrics.BalancedAccuracy
metrics.ClassificationReport
metrics.CohenKappa
metrics.Completeness
metrics.ConfusionMatrix
metrics.CrossEntropy
metrics.F1
metrics.FBeta
metrics.FowlkesMallows
metrics.GeometricMean
metrics.Homogeneity
metrics.Jaccard
metrics.LogLoss
metrics.MAE
metrics.MAPE
metrics.MCC
metrics.MSE
metrics.MacroF1
metrics.MacroFBeta
metrics.MacroJaccard
metrics.MacroPrecision
metrics.MacroRecall
metrics.MicroF1
metrics.MicroFBeta
metrics.MicroJaccard
metrics.MicroPrecision
metrics.MicroRecall
metrics.MultiFBeta
metrics.MutualInfo
metrics.NormalizedMutualInfo
metrics.Precision
metrics.R2
metrics.RMSE
metrics.RMSLE
metrics.ROCAUC
metrics.Rand
metrics.Recall
metrics.RollingROCAUC
metrics.SMAPE
metrics.Silhouette
metrics.VBeta
metrics.WeightedF1
metrics.WeightedFBeta
metrics.WeightedJaccard
metrics.WeightedPrecision
metrics.WeightedRecall
proba.Beta
proba.Gaussian
proba.Multinomial
proba.MultivariateGaussian
stats.BayesianMean
stats.Cov
stats.KolmogorovSmirnov
stats.Mean
stats.PearsonCorr
stats.SEM
stats.Sum
stats.Var
```
