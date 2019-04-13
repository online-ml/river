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

   compat.convert_creme_to_sklearn
   compat.convert_sklearn_to_creme
   compat.CremeClassifierWrapper
   compat.CremeRegressorWrapper
   compat.SKLClassifierWrapper
   compat.SKLClustererWrapper
   compat.SKLTransformerWrapper
   compat.SKLRegressorWrapper


:mod:`creme.compose`: Model composition
---------------------------------------

.. automodule:: creme.compose
   :no-members:
   :no-inherited-members:

.. currentmodule:: creme

.. autosummary::
   :toctree: generated/
   :nosignatures:

   compose.Blacklister
   compose.BoxCoxTransformRegressor
   compose.FuncTransformer
   compose.Pipeline
   compose.SplitRegressor
   compose.TargetModifierRegressor
   compose.TransformerUnion
   compose.Whitelister


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
   datasets.load_airline


:mod:`creme.dummy`: Dummy models
--------------------------------

.. automodule:: creme.dummy
   :no-members:
   :no-inherited-members:

.. currentmodule:: creme

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dummy.NoChangeClassifier
   dummy.PriorClassifier
   dummy.StatisticRegressor


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
   feature_extraction.TargetGroupBy
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
   feature_selection.SelectKBest
   feature_selection.VarianceThreshold


:mod:`creme.impute`: Running imputation
---------------------------------------

.. automodule:: creme.impute
   :no-members:
   :no-inherited-members:

.. currentmodule:: creme

.. autosummary::
   :toctree: generated/
   :nosignatures:

   impute.Imputer


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


:mod:`creme.metrics`: Streaming metrics
---------------------------------------

.. automodule:: creme.metrics
   :no-members:
   :no-inherited-members:

.. currentmodule:: creme

Binary classification
+++++++++++++++++++++

.. autosummary::
   :toctree: generated/
   :nosignatures:

   metrics.LogLoss
   metrics.F1Score
   metrics.Precision
   metrics.Recall

Multi-class classification
++++++++++++++++++++++++++

Note that every multi-class classification metric also works for binary classification. For example you may use the ``Accuracy`` metric in both cases.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   metrics.Accuracy
   metrics.ConfusionMatrix
   metrics.MacroF1Score
   metrics.MacroPrecision
   metrics.MacroRecall
   metrics.MicroF1Score
   metrics.MicroPrecision
   metrics.MicroRecall

Regression
++++++++++

.. autosummary::
   :toctree: generated/
   :nosignatures:

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

   naive_bayes.GaussianNB
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
   optim.Optimizer
   optim.PassiveAggressiveI
   optim.PassiveAggressiveII
   optim.RMSProp
   optim.VanillaSGD


Learning rate schedulers
++++++++++++++++++++++++

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optim.ConstantLR
   optim.InverseScalingLR


Loss functions
++++++++++++++

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optim.AbsoluteLoss
   optim.CauchyLoss
   optim.EpsilonInsensitiveHingeLoss
   optim.HingeLoss
   optim.LogLoss
   optim.Loss
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

   preprocessing.FeatureHasher
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

Univariate statistics
+++++++++++++++++++++

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
   stats.SEM
   stats.Skew
   stats.Sum
   stats.RollingMax
   stats.RollingMean
   stats.RollingMin
   stats.RollingMode
   stats.RollingPeakToPeak
   stats.RollingQuantile
   stats.RollingSEM
   stats.RollingSum
   stats.RollingVariance
   stats.Univariate
   stats.Variance

Bivariate statistics
++++++++++++++++++++

.. autosummary::
   :toctree: generated/
   :nosignatures:

   stats.Bivariate
   stats.Covariance
   stats.PearsonCorrelation


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


:mod:`creme.tree`: Incremental decision trees
-----------------------------------------------

.. automodule:: creme.tree
   :no-members:
   :no-inherited-members:

.. currentmodule:: creme

.. autosummary::
   :toctree: generated/
   :nosignatures:

   tree.MondrianTreeClassifier
   tree.MondrianTreeRegressor
