API reference
=============

:mod:`creme.base`: Base classes
-------------------------------

.. automodule:: creme.base
   :no-members:
   :no-inherited-members:

.. currentmodule:: creme

.. autosummary::
   :toctree: generated/
   :nosignatures:

   base.BinaryClassifier
   base.Clusterer
   base.MultiClassifier
   base.Regressor
   base.Transformer


:mod:`creme.cluster`: Clustering
--------------------------------

.. automodule:: creme.cluster
   :no-members:
   :no-inherited-members:

.. currentmodule:: creme

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cluster.KMeans


:mod:`creme.compat`: Compatibility utilities
--------------------------------------------

.. automodule:: creme.compat
   :no-members:
   :no-inherited-members:

.. currentmodule:: creme

.. autosummary::
   :toctree: generated/
   :nosignatures:

   compat.SKLClassifierWrapper
   compat.SKLClustererWrapper
   compat.SKLTransformerWrapper
   compat.SKLRegressorWrapper
   compat.wrap_sklearn


:mod:`creme.compose`: Model composition
---------------------------------------

.. automodule:: creme.compose
   :no-members:
   :no-inherited-members:

.. currentmodule:: creme

.. autosummary::
   :toctree: generated/
   :nosignatures:

   compose.BoxCoxTransformRegressor
   compose.Pipeline
   compose.TargetModifierRegressor
   compose.TransformerUnion


:mod:`creme.datasets`: Dataset loading utilities
------------------------------------------------

.. automodule:: creme.datasets
   :no-members:
   :no-inherited-members:

.. currentmodule:: creme

.. autosummary::
   :toctree: generated/
   :nosignatures:

   datasets.fetch_restaurants


:mod:`creme.ensemble`: Ensemble models
--------------------------------------

.. automodule:: creme.ensemble
   :no-members:
   :no-inherited-members:

.. currentmodule:: creme

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ensemble.BaggingClassifier
   ensemble.HedgeClassifier


:mod:`creme.feature_extraction`: Feature extraction
---------------------------------------------------

.. automodule:: creme.feature_extraction
   :no-members:
   :no-inherited-members:

.. currentmodule:: creme

.. autosummary::
   :toctree: generated/
   :nosignatures:

   feature_extraction.CountVectorizer
   feature_extraction.GroupBy
   feature_extraction.TargetEncoder
   feature_extraction.TFIDFVectorizer


:mod:`creme.feature_selection`: Feature selection
-------------------------------------------------

.. automodule:: creme.feature_selection
   :no-members:
   :no-inherited-members:

.. currentmodule:: creme

.. autosummary::
   :toctree: generated/
   :nosignatures:

   feature_selection.RandomDiscarder


:mod:`creme.impute`: Running imputation
---------------------------------------

.. automodule:: creme.impute
   :no-members:
   :no-inherited-members:

.. currentmodule:: creme

.. autosummary::
   :toctree: generated/
   :nosignatures:

   impute.NumericImputer
   impute.CategoricalImputer


:mod:`creme.linear_model`: Linear models
----------------------------------------

.. automodule:: creme.linear_model
   :no-members:
   :no-inherited-members:

.. currentmodule:: creme

.. autosummary::
   :toctree: generated/
   :nosignatures:

   linear_model.FMRegressor
   linear_model.LinearRegression
   linear_model.LogisticRegression
   linear_model.PassiveAggressiveClassifier
   linear_model.PassiveAggressiveRegressor
   linear_model.PassiveAggressiveIClassifier
   linear_model.PassiveAggressiveIRegressor
   linear_model.PassiveAggressiveIIClassifier
   linear_model.PassiveAggressiveIIRegressor


:mod:`creme.metrics`: Streaming metrics
---------------------------------------

.. automodule:: creme.metrics
   :no-members:
   :no-inherited-members:

.. currentmodule:: creme

.. autosummary::
   :toctree: generated/
   :nosignatures:

   metrics.Accuracy
   metrics.MAE
   metrics.MSE
   metrics.RMSE
   metrics.RMSLE
   metrics.SMAPE


:mod:`creme.model_selection`: Model selection
---------------------------------------------

.. automodule:: creme.model_selection
   :no-members:
   :no-inherited-members:

.. currentmodule:: creme

.. autosummary::
   :toctree: generated/
   :nosignatures:

   model_selection.online_score


:mod:`creme.multiclass`: Multi-class classification
---------------------------------------------------

.. automodule:: creme.multiclass
   :no-members:
   :no-inherited-members:

.. currentmodule:: creme

.. autosummary::
   :toctree: generated/
   :nosignatures:

   multiclass.OneVsRestClassifier


:mod:`creme.naive_bayes`: Naive Bayes models
--------------------------------------------

.. automodule:: creme.naive_bayes
   :no-members:
   :no-inherited-members:

.. currentmodule:: creme

.. autosummary::
   :toctree: generated/
   :nosignatures:

   naive_bayes.MultinomialNB


:mod:`creme.optim`: Optimization
--------------------------------

.. automodule:: creme.optim
   :no-members:
   :no-inherited-members:

.. currentmodule:: creme

Optimizers
++++++++++

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optim.AdaDelta
   optim.AdaGrad
   optim.Adam
   optim.FTRLProximal
   optim.Momentum
   optim.NesterovMomentum
   optim.RMSProp
   optim.VanillaSGD


Learning rate schedulers
++++++++++++++++++++++++

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optim.ConstantLR
   optim.LinearDecreaseLR


Loss functions
++++++++++++++

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optim.EpsilonInsensitiveHingeLoss
   optim.LogLoss
   optim.HingeLoss
   optim.AbsoluteLoss
   optim.SquaredLoss


:mod:`creme.preprocessing`: Preprocessing
-----------------------------------------

.. automodule:: creme.preprocessing
   :no-members:
   :no-inherited-members:

.. currentmodule:: creme

.. autosummary::
   :toctree: generated/
   :nosignatures:

   preprocessing.Discarder
   preprocessing.FeatureHasher
   preprocessing.FuncTransformer
   preprocessing.MinMaxScaler
   preprocessing.OneHotEncoder
   preprocessing.PolynomialExtender
   preprocessing.StandardScaler


:mod:`creme.reco`: Recommendation algorithms
--------------------------------------------

.. automodule:: creme.reco
   :no-members:
   :no-inherited-members:

.. currentmodule:: creme

.. autosummary::
   :toctree: generated/
   :nosignatures:

   reco.RandomNormal
   reco.SGDBaseline
   reco.SVD


:mod:`creme.stats`: Running statistics
--------------------------------------

.. automodule:: creme.stats
   :no-members:
   :no-inherited-members:

.. currentmodule:: creme

.. autosummary::
   :toctree: generated/
   :nosignatures:

   stats.Count
   stats.Entropy
   stats.EWMean
   stats.Kurtosis
   stats.Max
   stats.Mean
   stats.Min
   stats.Mode
   stats.NUnique
   stats.PeakToPeak
   stats.Quantile
   stats.Skew
   stats.Sum
   stats.RollingMax
   stats.RollingMean
   stats.RollingMin
   stats.RollingMode
   stats.RollingPeakToPeak
   stats.RollingQuantile
   stats.RollingSum
   stats.RollingVariance
   stats.Variance


:mod:`creme.stream`: Streaming utilities
----------------------------------------

.. automodule:: creme.stream
   :no-members:
   :no-inherited-members:

.. currentmodule:: creme

.. autosummary::
   :toctree: generated/
   :nosignatures:

   stream.iter_csv
   stream.iter_numpy
   stream.iter_sklearn_dataset
   stream.iter_pandas
