API reference
=============

**anomaly**: Anomaly detection
-------------------------------

The estimators in ``anomaly`` are slightly different than the rest of the estimators. Instead of a ``predict_one`` method, each anomaly detector has a ``score_one`` method which returns an anomaly score for a given set of features. Anomalies will have high scores whereas normal observations will have low scores. The range of the scores depends on the estimator. 

.. rubric:: Classes

.. currentmodule:: creme.anomaly
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        
    HalfSpaceTrees


**base**: Base interfaces
--------------------------

Every estimator in ``creme`` is a class, and as such inherits from at least one base interface. These are used to categorize, organize, and standardize the many estimators that ``creme`` contains. 

.. rubric:: Classes

.. currentmodule:: creme.base
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        
    AnomalyDetector
    BinaryClassifier
    Classifier
    Clusterer
    Ensemble
    Estimator
    MultiClassifier
    MultiOutputClassifier
    MultiOutputRegressor
    Regressor
    Transformer
    Wrapper


**cluster**: Unsupervised clustering
-------------------------------------

.. rubric:: Classes

.. currentmodule:: creme.cluster
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        
    KMeans


**compat**: Compatibility with other libraries
-----------------------------------------------

This module contains adapters for making ``creme`` estimators compatible with other libraries, and vice-versa whenever possible. 

.. rubric:: Classes

.. currentmodule:: creme.compat
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        
    Creme2SKLClassifier
    Creme2SKLClusterer
    Creme2SKLRegressor
    Creme2SKLTransformer
    PyTorch2CremeRegressor
    SKL2CremeClassifier
    SKL2CremeRegressor

.. rubric:: Functions

.. currentmodule:: creme.compat
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: function.rst
        
    convert_creme_to_sklearn
    convert_sklearn_to_creme


**compose**: Models composition
--------------------------------

.. rubric:: Classes

.. currentmodule:: creme.compose
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        
    Blacklister
    FuncTransformer
    Pipeline
    Renamer
    TransformerUnion
    Whitelister


**datasets**: Datasets
-----------------------

.. rubric:: Classes

.. currentmodule:: creme.datasets
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        
    Airline
    Bananas
    Bikes
    ChickWeights
    CreditCard
    Elec2
    Higgs
    ImageSegments
    KDD99HTTP
    MaliciousURL
    MovieLens100K
    Phishing
    Restaurants
    SMSSpam
    TREC07
    Taxis
    TrumpApproval

Random data generators
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        
    gen.SEA
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: function.rst
        


**decomposition**: Online matrix decomposition
-----------------------------------------------

.. rubric:: Classes

.. currentmodule:: creme.decomposition
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        
    LDA


**dummy**: Dummy estimators
----------------------------

.. rubric:: Classes

.. currentmodule:: creme.dummy
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        
    NoChangeClassifier
    PriorClassifier
    StatisticRegressor


**ensemble**: Ensemble learning
--------------------------------

.. rubric:: Classes

.. currentmodule:: creme.ensemble
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        
    AdaBoostClassifier
    BaggingClassifier
    BaggingRegressor
    HedgeRegressor
    StackingBinaryClassifier


**facto**: Factorization machines
----------------------------------

.. rubric:: Classes

.. currentmodule:: creme.facto
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        
    FFMClassifier
    FFMRegressor
    FMClassifier
    FMRegressor
    FwFMClassifier
    FwFMRegressor
    HOFMClassifier
    HOFMRegressor


**feature_extraction**: Feature extraction from a stream
---------------------------------------------------------

.. rubric:: Classes

.. currentmodule:: creme.feature_extraction
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        
    Agg
    BagOfWords
    Differ
    TFIDF
    TargetAgg


**feature_selection**: Online feature selection
------------------------------------------------

.. rubric:: Classes

.. currentmodule:: creme.feature_selection
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        
    PoissonInclusion
    SelectKBest
    VarianceThreshold


**impute**: Missing data imputation
------------------------------------

.. rubric:: Classes

.. currentmodule:: creme.impute
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        
    PreviousImputer
    StatImputer


**linear_model**: Linear models
--------------------------------

.. rubric:: Classes

.. currentmodule:: creme.linear_model
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        
    ALMAClassifier
    LinearRegression
    LogisticRegression
    PAClassifier
    PARegressor
    SoftmaxRegression


**meta**: Meta-models that wrap other models
---------------------------------------------

.. rubric:: Classes

.. currentmodule:: creme.meta
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        
    BoxCoxRegressor
    PredClipper
    TransformedTargetRegressor


**metrics**: Streaming metrics
-------------------------------

.. rubric:: Classes

.. currentmodule:: creme.metrics
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        
    Accuracy
    ClassificationReport
    ConfusionMatrix
    CrossEntropy
    F1
    FBeta
    Jaccard
    LogLoss
    MAE
    MCC
    MSE
    MacroF1
    MacroFBeta
    MacroPrecision
    MacroRecall
    Metric
    MicroF1
    MicroFBeta
    MicroPrecision
    MicroRecall
    MultiFBeta
    Precision
    RMSE
    RMSLE
    ROCAUC
    Recall
    RegressionMultiOutput
    Rolling
    SMAPE
    TimeRolling
    WeightedF1
    WeightedFBeta
    WeightedPrecision
    WeightedRecall


**model_selection**: Model selection and evaluation
----------------------------------------------------

.. rubric:: Functions

.. currentmodule:: creme.model_selection
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: function.rst
        
    expand_param_grid
    progressive_val_score
    successive_halving


**multiclass**: Multi-class classification
-------------------------------------------

.. rubric:: Classes

.. currentmodule:: creme.multiclass
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        
    OneVsRestClassifier


**multioutput**: Multi-output classification and regression
------------------------------------------------------------

.. rubric:: Classes

.. currentmodule:: creme.multioutput
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        
    ClassifierChain
    RegressorChain


**naive_bayes**: Naive Bayes algorithms
----------------------------------------

.. rubric:: Classes

.. currentmodule:: creme.naive_bayes
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        
    BernoulliNB
    ComplementNB
    GaussianNB
    MultinomialNB


**neighbors**: Neighbors-based learning
----------------------------------------

.. rubric:: Classes

.. currentmodule:: creme.neighbors
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        
    KNeighborsClassifier
    KNeighborsRegressor


**optim**: Online optimization
-------------------------------

.. rubric:: Classes

.. currentmodule:: creme.optim
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        
    AMSGrad
    AdaBound
    AdaDelta
    AdaGrad
    AdaMax
    Adam
    FTRLProximal
    MiniBatcher
    Momentum
    Nadam
    NesterovMomentum
    Optimizer
    RMSProp
    SGD

Weight initialization schemes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        
    initializers.Constant
    initializers.Normal
    initializers.Zeros
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: function.rst
        

Loss functions
~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        
    losses.Absolute
    losses.BinaryFocalLoss
    losses.Cauchy
    losses.CrossEntropy
    losses.EpsilonInsensitiveHinge
    losses.Hinge
    losses.Log
    losses.Perceptron
    losses.Poisson
    losses.Quantile
    losses.Squared
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: function.rst
        

Learning rate schedulers
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        
    schedulers.Constant
    schedulers.InverseScaling
    schedulers.Optimal
    schedulers.Scheduler
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: function.rst
        


**preprocessing**: Feature preprocessing
-----------------------------------------

.. rubric:: Classes

.. currentmodule:: creme.preprocessing
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        
    Binarizer
    FeatureHasher
    MaxAbsScaler
    MinMaxScaler
    Normalizer
    OneHotEncoder
    PolynomialExtender
    RBFSampler
    RobustScaler
    StandardScaler


**proba**: Probability distributions
-------------------------------------

.. rubric:: Classes

.. currentmodule:: creme.proba
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        
    Gaussian
    Multinomial


**reco**: Recommender systems
------------------------------

.. rubric:: Classes

.. currentmodule:: creme.reco
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        
    Baseline
    BiasedMF
    FunkMF
    RandomNormal


**sampling**: Sampling methods
-------------------------------

.. rubric:: Classes

.. currentmodule:: creme.sampling
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        
    HardSamplingClassifier
    HardSamplingRegressor
    RandomOverSampler
    RandomSampler
    RandomUnderSampler


**stats**: Running statistics
------------------------------

.. rubric:: Classes

.. currentmodule:: creme.stats
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        
    AbsMax
    AutoCorrelation
    BayesianMean
    Bivariate
    Count
    Covariance
    EWMean
    EWVar
    Entropy
    IQR
    Kurtosis
    Max
    Mean
    Min
    Mode
    NUnique
    PeakToPeak
    PearsonCorrelation
    Quantile
    RollingAbsMax
    RollingIQR
    RollingMax
    RollingMean
    RollingMin
    RollingMode
    RollingPeakToPeak
    RollingQuantile
    RollingSEM
    RollingSum
    RollingVar
    SEM
    Skew
    Sum
    Univariate
    Var


**stream**: Utilities for handling streaming datasets
------------------------------------------------------

.. rubric:: Classes

.. currentmodule:: creme.stream
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        
    Cache

.. rubric:: Functions

.. currentmodule:: creme.stream
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: function.rst
        
    iter_array
    iter_csv
    iter_libsvm
    iter_pandas
    iter_sklearn_dataset
    iter_vaex
    shuffle
    simulate_qa


**time_series**: Time series forecasting
-----------------------------------------

.. rubric:: Classes

.. currentmodule:: creme.time_series
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        
    Detrender
    GroupDetrender
    SNARIMAX


**tree**: Decision trees
-------------------------

.. rubric:: Classes

.. currentmodule:: creme.tree
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        
    DecisionTreeClassifier
    RandomForestClassifier


**utils**: Utility classes and functions
-----------------------------------------

.. rubric:: Classes

.. currentmodule:: creme.utils
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        
    Histogram
    SDFT
    Skyline
    SortedWindow
    Window

Utilities for unit testing and sanity checking estimators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: function.rst
        
    estimator_checks.check_estimator
    estimator_checks.guess_model

Mathematical utility functions (intended for internal purposes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A lot of this is experimental and has a high probability of changing in the future. 

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: function.rst
        
    math.chain_dot
    math.clamp
    math.dot
    math.dotvecmat
    math.matmul2d
    math.minkowski_distance
    math.norm
    math.outer
    math.prod
    math.sherman_morrison
    math.sigmoid
    math.sign
    math.softmax

Helper functions for making things readable by humans
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: function.rst
        
    pretty.format_object
    pretty.print_table


