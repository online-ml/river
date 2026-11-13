# Overview

## active

Online active learning.

- [EntropySampler](<https://riverml.xyz/0.27.0/api/active/EntropySampler>)

### base

- [ActiveLearningClassifier](<https://riverml.xyz/0.27.0/api/active/base/ActiveLearningClassifier>)

## anomaly

Anomaly detection.

Estimators in the `anomaly` module have a bespoke API. Each anomaly detector has a `score_one` method instead of a `predict_one` method. This method returns an anomaly score. Normal observations should have a low score, whereas anomalous observations should have a high score. The range of the scores is relative to each estimator.

Anomaly detectors are usually unsupervised, in that they analyze the distribution of the features they are shown. But River also has a notion of supervised anomaly detectors. These analyze the distribution of a target variable, and optionally include the distribution of the features as well. They are useful for detecting labelling anomalies, which can be detrimental if they learned by a model.

- [GaussianScorer](<https://riverml.xyz/0.27.0/api/anomaly/GaussianScorer>)
- [HalfSpaceTrees](<https://riverml.xyz/0.27.0/api/anomaly/HalfSpaceTrees>)
- [LODA](<https://riverml.xyz/0.27.0/api/anomaly/LODA>)
- [LocalOutlierFactor](<https://riverml.xyz/0.27.0/api/anomaly/LocalOutlierFactor>)
- [OneClassSVM](<https://riverml.xyz/0.27.0/api/anomaly/OneClassSVM>)
- [PredictiveAnomalyDetection](<https://riverml.xyz/0.27.0/api/anomaly/PredictiveAnomalyDetection>)
- [QuantileFilter](<https://riverml.xyz/0.27.0/api/anomaly/QuantileFilter>)
- [StandardAbsoluteDeviation](<https://riverml.xyz/0.27.0/api/anomaly/StandardAbsoluteDeviation>)
- [ThresholdFilter](<https://riverml.xyz/0.27.0/api/anomaly/ThresholdFilter>)

## bandit

Multi-armed bandit (MAB) policies.

The bandit policies in River have a generic API. This allows them to be used in a variety of situations. For instance, they can be used for model selection (see [`model_selection.BanditRegressor`](<https://riverml.xyz/0.27.0/api/model-selection/BanditRegressor>)).

**Classes**

- [BayesUCB](<https://riverml.xyz/0.27.0/api/bandit/BayesUCB>)
- [EpsilonGreedy](<https://riverml.xyz/0.27.0/api/bandit/EpsilonGreedy>)
- [Exp3](<https://riverml.xyz/0.27.0/api/bandit/Exp3>)
- [LinUCBDisjoint](<https://riverml.xyz/0.27.0/api/bandit/LinUCBDisjoint>)
- [RandomPolicy](<https://riverml.xyz/0.27.0/api/bandit/RandomPolicy>)
- [ThompsonSampling](<https://riverml.xyz/0.27.0/api/bandit/ThompsonSampling>)
- [UCB](<https://riverml.xyz/0.27.0/api/bandit/UCB>)

**Functions**

- [evaluate](<https://riverml.xyz/0.27.0/api/bandit/evaluate>)
- [evaluate\_offline](<https://riverml.xyz/0.27.0/api/bandit/evaluate-offline>)

### base

- [ContextualPolicy](<https://riverml.xyz/0.27.0/api/bandit/base/ContextualPolicy>)
- [Policy](<https://riverml.xyz/0.27.0/api/bandit/base/Policy>)

### datasets

- [BanditDataset](<https://riverml.xyz/0.27.0/api/bandit/datasets/BanditDataset>)
- [NewsArticles](<https://riverml.xyz/0.27.0/api/bandit/datasets/NewsArticles>)

### envs

- [CandyCaneContest](<https://riverml.xyz/0.27.0/api/bandit/envs/CandyCaneContest>)
- [KArmedTestbed](<https://riverml.xyz/0.27.0/api/bandit/envs/KArmedTestbed>)

## base

Base interfaces.

Every estimator in River is a class, and as such inherits from at least one base interface. These are used to categorize, organize, and standardize the many estimators that River contains.

This module contains mixin classes, which are all suffixed by `Mixin`. Their purpose is to provide additional functionality to an estimator, and thus need to be used in conjunction with a non-mixin base class.

This module also contains utilities for type hinting and tagging estimators.

- [AnomalyDetector](<https://riverml.xyz/0.27.0/api/base/AnomalyDetector>)
- [AnomalyFilter](<https://riverml.xyz/0.27.0/api/base/AnomalyFilter>)
- [Base](<https://riverml.xyz/0.27.0/api/base/Base>)
- [BaseTransformer](<https://riverml.xyz/0.27.0/api/base/BaseTransformer>)
- [BinaryDriftAndWarningDetector](<https://riverml.xyz/0.27.0/api/base/BinaryDriftAndWarningDetector>)
- [BinaryDriftDetector](<https://riverml.xyz/0.27.0/api/base/BinaryDriftDetector>)
- [Classifier](<https://riverml.xyz/0.27.0/api/base/Classifier>)
- [Clusterer](<https://riverml.xyz/0.27.0/api/base/Clusterer>)
- [DriftAndWarningDetector](<https://riverml.xyz/0.27.0/api/base/DriftAndWarningDetector>)
- [DriftDetector](<https://riverml.xyz/0.27.0/api/base/DriftDetector>)
- [Ensemble](<https://riverml.xyz/0.27.0/api/base/Ensemble>)
- [Estimator](<https://riverml.xyz/0.27.0/api/base/Estimator>)
- [MiniBatchClassifier](<https://riverml.xyz/0.27.0/api/base/MiniBatchClassifier>)
- [MiniBatchRegressor](<https://riverml.xyz/0.27.0/api/base/MiniBatchRegressor>)
- [MiniBatchSupervisedTransformer](<https://riverml.xyz/0.27.0/api/base/MiniBatchSupervisedTransformer>)
- [MiniBatchTransformer](<https://riverml.xyz/0.27.0/api/base/MiniBatchTransformer>)
- [MultiLabelClassifier](<https://riverml.xyz/0.27.0/api/base/MultiLabelClassifier>)
- [MultiTargetRegressor](<https://riverml.xyz/0.27.0/api/base/MultiTargetRegressor>)
- [Regressor](<https://riverml.xyz/0.27.0/api/base/Regressor>)
- [SupervisedAnomalyDetector](<https://riverml.xyz/0.27.0/api/base/SupervisedAnomalyDetector>)
- [SupervisedTransformer](<https://riverml.xyz/0.27.0/api/base/SupervisedTransformer>)
- [Transformer](<https://riverml.xyz/0.27.0/api/base/Transformer>)
- [Wrapper](<https://riverml.xyz/0.27.0/api/base/Wrapper>)
- [WrapperEnsemble](<https://riverml.xyz/0.27.0/api/base/WrapperEnsemble>)

## cluster

Unsupervised clustering.

- [CluStream](<https://riverml.xyz/0.27.0/api/cluster/CluStream>)
- [DBSTREAM](<https://riverml.xyz/0.27.0/api/cluster/DBSTREAM>)
- [DenStream](<https://riverml.xyz/0.27.0/api/cluster/DenStream>)
- [KMeans](<https://riverml.xyz/0.27.0/api/cluster/KMeans>)
- [ODAC](<https://riverml.xyz/0.27.0/api/cluster/ODAC>)
- [STREAMKMeans](<https://riverml.xyz/0.27.0/api/cluster/STREAMKMeans>)
- [TextClust](<https://riverml.xyz/0.27.0/api/cluster/TextClust>)

## compat

Compatibility tools.

This module contains adapters for making River estimators compatible with other libraries, and vice-versa whenever possible. The relevant adapters will only be usable if you have installed the necessary library. For instance, you have to install scikit-learn in order to use the [`compat.convert_sklearn_to_river`](<https://riverml.xyz/0.27.0/api/compat/convert-sklearn-to-river>) function.

**Classes**

- [River2SKLClassifier](<https://riverml.xyz/0.27.0/api/compat/River2SKLClassifier>)
- [River2SKLClusterer](<https://riverml.xyz/0.27.0/api/compat/River2SKLClusterer>)
- [River2SKLRegressor](<https://riverml.xyz/0.27.0/api/compat/River2SKLRegressor>)
- [River2SKLTransformer](<https://riverml.xyz/0.27.0/api/compat/River2SKLTransformer>)
- [SKL2RiverClassifier](<https://riverml.xyz/0.27.0/api/compat/SKL2RiverClassifier>)
- [SKL2RiverRegressor](<https://riverml.xyz/0.27.0/api/compat/SKL2RiverRegressor>)

**Functions**

- [convert\_river\_to\_sklearn](<https://riverml.xyz/0.27.0/api/compat/convert-river-to-sklearn>)
- [convert\_sklearn\_to\_river](<https://riverml.xyz/0.27.0/api/compat/convert-sklearn-to-river>)

## compose

Model composition.

This module contains utilities for merging multiple modeling steps into a single pipeline. Although pipelines are not the only way to process a stream of data, we highly encourage you to use them.

**Classes**

- [Discard](<https://riverml.xyz/0.27.0/api/compose/Discard>)
- [FuncTransformer](<https://riverml.xyz/0.27.0/api/compose/FuncTransformer>)
- [Grouper](<https://riverml.xyz/0.27.0/api/compose/Grouper>)
- [Pipeline](<https://riverml.xyz/0.27.0/api/compose/Pipeline>)
- [Prefixer](<https://riverml.xyz/0.27.0/api/compose/Prefixer>)
- [Renamer](<https://riverml.xyz/0.27.0/api/compose/Renamer>)
- [Select](<https://riverml.xyz/0.27.0/api/compose/Select>)
- [SelectType](<https://riverml.xyz/0.27.0/api/compose/SelectType>)
- [Suffixer](<https://riverml.xyz/0.27.0/api/compose/Suffixer>)
- [TargetTransformRegressor](<https://riverml.xyz/0.27.0/api/compose/TargetTransformRegressor>)
- [TransformerProduct](<https://riverml.xyz/0.27.0/api/compose/TransformerProduct>)
- [TransformerUnion](<https://riverml.xyz/0.27.0/api/compose/TransformerUnion>)

**Functions**

- [learn\_during\_predict](<https://riverml.xyz/0.27.0/api/compose/learn-during-predict>)

## conf

Conformal predictions.

This modules contains wrappers to enable conformal predictions on any regressor or classifier.

- [Interval](<https://riverml.xyz/0.27.0/api/conf/Interval>)
- [RegressionJackknife](<https://riverml.xyz/0.27.0/api/conf/RegressionJackknife>)

## covariance

Online estimation of covariance and precision matrices.

A covariance matrix summarises how a set of variables move together. It is the engine behind portfolio risk, anomaly detection (via the Mahalanobis distance), Gaussian models, and many dimensionality-reduction methods. This module estimates it (and its inverse, the precision matrix) incrementally from a stream, without storing the data. See each estimator's docstring for what it does and when to reach for it.

The estimators are dict-native: `update(x)` takes a mapping and the `matrix` is a dict of pairwise values. Most also expose an `update_many` method for mini-batches of any narwhals-compatible dataframe.

- [EmpiricalCovariance](<https://riverml.xyz/0.27.0/api/covariance/EmpiricalCovariance>)
- [EmpiricalPrecision](<https://riverml.xyz/0.27.0/api/covariance/EmpiricalPrecision>)
- [EwaCovariance](<https://riverml.xyz/0.27.0/api/covariance/EwaCovariance>)
- [EwaPrecision](<https://riverml.xyz/0.27.0/api/covariance/EwaPrecision>)
- [LedoitWolfCovariance](<https://riverml.xyz/0.27.0/api/covariance/LedoitWolfCovariance>)
- [OASCovariance](<https://riverml.xyz/0.27.0/api/covariance/OASCovariance>)
- [ShrunkCovariance](<https://riverml.xyz/0.27.0/api/covariance/ShrunkCovariance>)

## datasets

Datasets.

This module contains a collection of datasets for multiple tasks: classification, regression, etc. The data corresponds to popular datasets and are conveniently wrapped to easily iterate over the data in a stream fashion. All datasets have fixed size. Please refer to `river.synth` if you are interested in infinite synthetic data generators.

**Regression**

| Name | Samples | Features |
| :--- | ---: | ---: |
| [AirlinePassengers](<https://riverml.xyz/0.27.0/api/datasets/AirlinePassengers>) | 144 | 1 |
| [Bikes](<https://riverml.xyz/0.27.0/api/datasets/Bikes>) | 182,470 | 8 |
| [ChickWeights](<https://riverml.xyz/0.27.0/api/datasets/ChickWeights>) | 578 | 3 |
| [MovieLens100K](<https://riverml.xyz/0.27.0/api/datasets/MovieLens100K>) | 100,000 | 10 |
| [Restaurants](<https://riverml.xyz/0.27.0/api/datasets/Restaurants>) | 252,108 | 7 |
| [SP500Stocks](<https://riverml.xyz/0.27.0/api/datasets/SP500Stocks>) | 1,257 | 10 |
| [Taxis](<https://riverml.xyz/0.27.0/api/datasets/Taxis>) | 1,458,644 | 8 |
| [TrumpApproval](<https://riverml.xyz/0.27.0/api/datasets/TrumpApproval>) | 1,001 | 6 |
| [WaterFlow](<https://riverml.xyz/0.27.0/api/datasets/WaterFlow>) | 1,268 | 1 |

**Binary classification**

| Name | Samples | Features | Sparse |
| :--- | ---: | ---: | :--- |
| [Bananas](<https://riverml.xyz/0.27.0/api/datasets/Bananas>) | 5,300 | 2 |  |
| [BETH](<https://riverml.xyz/0.27.0/api/datasets/BETH>) | 2,666,118 | 11 |  |
| [CreditCard](<https://riverml.xyz/0.27.0/api/datasets/CreditCard>) | 284,807 | 30 |  |
| [CriteoAds](<https://riverml.xyz/0.27.0/api/datasets/CriteoAds>) | 100,000 | 39 |  |
| [Elec2](<https://riverml.xyz/0.27.0/api/datasets/Elec2>) | 45,312 | 8 |  |
| [Higgs](<https://riverml.xyz/0.27.0/api/datasets/Higgs>) | 11,000,000 | 28 |  |
| [HTTP](<https://riverml.xyz/0.27.0/api/datasets/HTTP>) | 567,498 | 3 |  |
| [MaliciousURL](<https://riverml.xyz/0.27.0/api/datasets/MaliciousURL>) | 2,396,130 | 3,231,961 | ✔️ |
| [Phishing](<https://riverml.xyz/0.27.0/api/datasets/Phishing>) | 1,250 | 9 |  |
| [Shuttle](<https://riverml.xyz/0.27.0/api/datasets/Shuttle>) | 49,097 | 9 |  |
| [SMSSpam](<https://riverml.xyz/0.27.0/api/datasets/SMSSpam>) | 5,574 | 1 |  |
| [SMTP](<https://riverml.xyz/0.27.0/api/datasets/SMTP>) | 95,156 | 3 |  |
| [TREC07](<https://riverml.xyz/0.27.0/api/datasets/TREC07>) | 75,419 | 5 |  |

**Multi-class classification**

| Name | Samples | Features | Classes |
| :--- | ---: | ---: | ---: |
| [ImageSegments](<https://riverml.xyz/0.27.0/api/datasets/ImageSegments>) | 2,310 | 18 | 7 |
| [Insects](<https://riverml.xyz/0.27.0/api/datasets/Insects>) | 52,848 | 33 | 6 |
| [Keystroke](<https://riverml.xyz/0.27.0/api/datasets/Keystroke>) | 20,400 | 31 | 51 |
| [NewsT](<https://riverml.xyz/0.27.0/api/datasets/NewsT>) | 11,109 | 1 | 152 |
| [TwitchChat](<https://riverml.xyz/0.27.0/api/datasets/TwitchChat>) | 14,720,748 | 11 | 20 |

**Multi-output binary classification**

| Name | Samples | Features | Outputs |
| :--- | ---: | ---: | ---: |
| [Music](<https://riverml.xyz/0.27.0/api/datasets/Music>) | 593 | 72 | 6 |
| [Yeast](<https://riverml.xyz/0.27.0/api/datasets/Yeast>) | 2,417 | 103 | 14 |

**Multi-output regression**

| Name | Samples | Features | Outputs |
| :--- | ---: | ---: | ---: |
| [SolarFlare](<https://riverml.xyz/0.27.0/api/datasets/SolarFlare>) | 1,066 | 10 | 3 |
| [WebTraffic](<https://riverml.xyz/0.27.0/api/datasets/WebTraffic>) | 44,160 | 3 | 2 |

### base

- [Dataset](<https://riverml.xyz/0.27.0/api/datasets/base/Dataset>)
- [FileDataset](<https://riverml.xyz/0.27.0/api/datasets/base/FileDataset>)
- [RemoteDataset](<https://riverml.xyz/0.27.0/api/datasets/base/RemoteDataset>)
- [SyntheticDataset](<https://riverml.xyz/0.27.0/api/datasets/base/SyntheticDataset>)

### synth

Synthetic datasets.

Each synthetic dataset is a stream generator. The benefit of using a generator is that they do not store the data and each data sample is generated on the fly. Except for a couple of methods, the majority of these methods are infinite data generators.

**Binary classification**

| Name | Features |
| :--- | ---: |
| [Agrawal](<https://riverml.xyz/0.27.0/api/datasets/synth/Agrawal>) | 9 |
| [AnomalySine](<https://riverml.xyz/0.27.0/api/datasets/synth/AnomalySine>) | 2 |
| [ConceptDriftStream](<https://riverml.xyz/0.27.0/api/datasets/synth/ConceptDriftStream>) | 9 |
| [Hyperplane](<https://riverml.xyz/0.27.0/api/datasets/synth/Hyperplane>) | 10 |
| [Mixed](<https://riverml.xyz/0.27.0/api/datasets/synth/Mixed>) | 4 |
| [SEA](<https://riverml.xyz/0.27.0/api/datasets/synth/SEA>) | 3 |
| [Sine](<https://riverml.xyz/0.27.0/api/datasets/synth/Sine>) | 2 |
| [STAGGER](<https://riverml.xyz/0.27.0/api/datasets/synth/STAGGER>) | 3 |

**Regression**

| Name | Features |
| :--- | ---: |
| [Friedman](<https://riverml.xyz/0.27.0/api/datasets/synth/Friedman>) | 10 |
| [FriedmanDrift](<https://riverml.xyz/0.27.0/api/datasets/synth/FriedmanDrift>) | 10 |
| [Mv](<https://riverml.xyz/0.27.0/api/datasets/synth/Mv>) | 10 |
| [Planes2D](<https://riverml.xyz/0.27.0/api/datasets/synth/Planes2D>) | 10 |

**Multi-class classification**

| Name | Features | Classes |
| :--- | ---: | ---: |
| [LED](<https://riverml.xyz/0.27.0/api/datasets/synth/LED>) | 7 | 10 |
| [LEDDrift](<https://riverml.xyz/0.27.0/api/datasets/synth/LEDDrift>) | 7 | 10 |
| [RandomRBF](<https://riverml.xyz/0.27.0/api/datasets/synth/RandomRBF>) | 10 | 2 |
| [RandomRBFDrift](<https://riverml.xyz/0.27.0/api/datasets/synth/RandomRBFDrift>) | 10 | 2 |
| [RandomTree](<https://riverml.xyz/0.27.0/api/datasets/synth/RandomTree>) | 10 | 2 |
| [Waveform](<https://riverml.xyz/0.27.0/api/datasets/synth/Waveform>) | 21 | 3 |

**Multi-output binary classification**

| Name | Features | Outputs |
| :--- | ---: | ---: |
| [Logical](<https://riverml.xyz/0.27.0/api/datasets/synth/Logical>) | 2 | 3 |

## drift

Concept Drift Detection.

This module contains concept drift detection methods. The purpose of a drift detector is to raise an alarm if the data distribution changes. A good drift detector method is the one that maximizes the true positives while keeping the number of false positives to a minimum.

- [ADWIN](<https://riverml.xyz/0.27.0/api/drift/ADWIN>)
- [DriftRetrainingClassifier](<https://riverml.xyz/0.27.0/api/drift/DriftRetrainingClassifier>)
- [DummyDriftDetector](<https://riverml.xyz/0.27.0/api/drift/DummyDriftDetector>)
- [KSWIN](<https://riverml.xyz/0.27.0/api/drift/KSWIN>)
- [NoDrift](<https://riverml.xyz/0.27.0/api/drift/NoDrift>)
- [PageHinkley](<https://riverml.xyz/0.27.0/api/drift/PageHinkley>)

### binary

Drift detection for binary data.

- [DDM](<https://riverml.xyz/0.27.0/api/drift/binary/DDM>)
- [EDDM](<https://riverml.xyz/0.27.0/api/drift/binary/EDDM>)
- [FHDDM](<https://riverml.xyz/0.27.0/api/drift/binary/FHDDM>)
- [HDDMA](<https://riverml.xyz/0.27.0/api/drift/binary/HDDMA>)
- [HDDMW](<https://riverml.xyz/0.27.0/api/drift/binary/HDDMW>)

### datasets

- [AirlinePassengers](<https://riverml.xyz/0.27.0/api/drift/datasets/AirlinePassengers>)
- [Apple](<https://riverml.xyz/0.27.0/api/drift/datasets/Apple>)
- [Bitcoin](<https://riverml.xyz/0.27.0/api/drift/datasets/Bitcoin>)
- [BrentSpotPrice](<https://riverml.xyz/0.27.0/api/drift/datasets/BrentSpotPrice>)
- [Occupancy](<https://riverml.xyz/0.27.0/api/drift/datasets/Occupancy>)
- [RunLog](<https://riverml.xyz/0.27.0/api/drift/datasets/RunLog>)
- [UKCoalEmploy](<https://riverml.xyz/0.27.0/api/drift/datasets/UKCoalEmploy>)

## dummy

Dummy estimators.

This module is here for testing purposes, as well as providing baseline performances.

- [NoChangeClassifier](<https://riverml.xyz/0.27.0/api/dummy/NoChangeClassifier>)
- [PriorClassifier](<https://riverml.xyz/0.27.0/api/dummy/PriorClassifier>)
- [StatisticRegressor](<https://riverml.xyz/0.27.0/api/dummy/StatisticRegressor>)

## ensemble

Ensemble learning.

Broadly speaking, there are two kinds of ensemble approaches. There are those that copy a single model several times and aggregate the predictions of said copies. This includes bagging as well as boosting. Then there are those that are composed of an arbitrary list of models, and can therefore aggregate predictions from different kinds of models.

- [ADWINBaggingClassifier](<https://riverml.xyz/0.27.0/api/ensemble/ADWINBaggingClassifier>)
- [ADWINBoostingClassifier](<https://riverml.xyz/0.27.0/api/ensemble/ADWINBoostingClassifier>)
- [AdaBoostClassifier](<https://riverml.xyz/0.27.0/api/ensemble/AdaBoostClassifier>)
- [BOLEClassifier](<https://riverml.xyz/0.27.0/api/ensemble/BOLEClassifier>)
- [BaggingClassifier](<https://riverml.xyz/0.27.0/api/ensemble/BaggingClassifier>)
- [BaggingRegressor](<https://riverml.xyz/0.27.0/api/ensemble/BaggingRegressor>)
- [EWARegressor](<https://riverml.xyz/0.27.0/api/ensemble/EWARegressor>)
- [LeveragingBaggingClassifier](<https://riverml.xyz/0.27.0/api/ensemble/LeveragingBaggingClassifier>)
- [SRPClassifier](<https://riverml.xyz/0.27.0/api/ensemble/SRPClassifier>)
- [SRPRegressor](<https://riverml.xyz/0.27.0/api/ensemble/SRPRegressor>)
- [StackingClassifier](<https://riverml.xyz/0.27.0/api/ensemble/StackingClassifier>)
- [VotingClassifier](<https://riverml.xyz/0.27.0/api/ensemble/VotingClassifier>)

## evaluate

Model evaluation.

This module provides utilities to evaluate an online model. The goal is to reproduce a real-world scenario with high fidelity. The core function of this module is `progressive_val_score`, which allows to evaluate a model via progressive validation.

This module also exposes "tracks". A track is a predefined combination of a dataset and one or more metrics. This allows a principled manner to compare models with each other. For instance, the `RegressionTrack` contains several datasets and metrics to evaluate regression models. There is also a bare `Track` class to implement a custom track. The `benchmarks` directory at the root of the River repository uses these tracks.

**Classes**

- [BinaryClassificationTrack](<https://riverml.xyz/0.27.0/api/evaluate/BinaryClassificationTrack>)
- [ForecastingTrack](<https://riverml.xyz/0.27.0/api/evaluate/ForecastingTrack>)
- [MultiClassClassificationTrack](<https://riverml.xyz/0.27.0/api/evaluate/MultiClassClassificationTrack>)
- [RegressionTrack](<https://riverml.xyz/0.27.0/api/evaluate/RegressionTrack>)
- [Track](<https://riverml.xyz/0.27.0/api/evaluate/Track>)

**Functions**

- [evaluate](<https://riverml.xyz/0.27.0/api/evaluate/evaluate>)
- [iter\_evaluate](<https://riverml.xyz/0.27.0/api/evaluate/iter-evaluate>)
- [iter\_progressive\_val\_score](<https://riverml.xyz/0.27.0/api/evaluate/iter-progressive-val-score>)
- [progressive\_val\_score](<https://riverml.xyz/0.27.0/api/evaluate/progressive-val-score>)

## facto

Factorization machines.

- [FFMClassifier](<https://riverml.xyz/0.27.0/api/facto/FFMClassifier>)
- [FFMRegressor](<https://riverml.xyz/0.27.0/api/facto/FFMRegressor>)
- [FMClassifier](<https://riverml.xyz/0.27.0/api/facto/FMClassifier>)
- [FMRegressor](<https://riverml.xyz/0.27.0/api/facto/FMRegressor>)
- [FwFMClassifier](<https://riverml.xyz/0.27.0/api/facto/FwFMClassifier>)
- [FwFMRegressor](<https://riverml.xyz/0.27.0/api/facto/FwFMRegressor>)
- [HOFMClassifier](<https://riverml.xyz/0.27.0/api/facto/HOFMClassifier>)
- [HOFMRegressor](<https://riverml.xyz/0.27.0/api/facto/HOFMRegressor>)

## feature\_extraction

Feature extraction.

This module can be used to extract information from raw features. This includes encoding categorical data as well as looking at interactions between existing features. This differs from the `preprocessing` module, in that the latter's purpose is rather to clean the data so that it may be processed by a particular machine learning algorithm.

- [Agg](<https://riverml.xyz/0.27.0/api/feature-extraction/Agg>)
- [BagOfWords](<https://riverml.xyz/0.27.0/api/feature-extraction/BagOfWords>)
- [PolynomialExtender](<https://riverml.xyz/0.27.0/api/feature-extraction/PolynomialExtender>)
- [RBFSampler](<https://riverml.xyz/0.27.0/api/feature-extraction/RBFSampler>)
- [RandomTreesEmbedding](<https://riverml.xyz/0.27.0/api/feature-extraction/RandomTreesEmbedding>)
- [TFIDF](<https://riverml.xyz/0.27.0/api/feature-extraction/TFIDF>)
- [TargetAgg](<https://riverml.xyz/0.27.0/api/feature-extraction/TargetAgg>)

## feature\_selection

Feature selection.

- [PoissonInclusion](<https://riverml.xyz/0.27.0/api/feature-selection/PoissonInclusion>)
- [SelectKBest](<https://riverml.xyz/0.27.0/api/feature-selection/SelectKBest>)
- [VarianceThreshold](<https://riverml.xyz/0.27.0/api/feature-selection/VarianceThreshold>)

## forest

Implement forest-based classifiers and regressors.

- [AMFClassifier](<https://riverml.xyz/0.27.0/api/forest/AMFClassifier>)
- [AMFRegressor](<https://riverml.xyz/0.27.0/api/forest/AMFRegressor>)
- [ARFClassifier](<https://riverml.xyz/0.27.0/api/forest/ARFClassifier>)
- [ARFRegressor](<https://riverml.xyz/0.27.0/api/forest/ARFRegressor>)
- [OXTRegressor](<https://riverml.xyz/0.27.0/api/forest/OXTRegressor>)

## imblearn

Sampling methods.

- [ChebyshevOverSampler](<https://riverml.xyz/0.27.0/api/imblearn/ChebyshevOverSampler>)
- [ChebyshevUnderSampler](<https://riverml.xyz/0.27.0/api/imblearn/ChebyshevUnderSampler>)
- [HardSamplingClassifier](<https://riverml.xyz/0.27.0/api/imblearn/HardSamplingClassifier>)
- [HardSamplingRegressor](<https://riverml.xyz/0.27.0/api/imblearn/HardSamplingRegressor>)
- [RandomOverSampler](<https://riverml.xyz/0.27.0/api/imblearn/RandomOverSampler>)
- [RandomSampler](<https://riverml.xyz/0.27.0/api/imblearn/RandomSampler>)
- [RandomUnderSampler](<https://riverml.xyz/0.27.0/api/imblearn/RandomUnderSampler>)

## linear\_model

Linear models.

- [ALMAClassifier](<https://riverml.xyz/0.27.0/api/linear-model/ALMAClassifier>)
- [AdPredictor](<https://riverml.xyz/0.27.0/api/linear-model/AdPredictor>)
- [BayesianLinearRegression](<https://riverml.xyz/0.27.0/api/linear-model/BayesianLinearRegression>)
- [LinearRegression](<https://riverml.xyz/0.27.0/api/linear-model/LinearRegression>)
- [LogisticRegression](<https://riverml.xyz/0.27.0/api/linear-model/LogisticRegression>)
- [PAClassifier](<https://riverml.xyz/0.27.0/api/linear-model/PAClassifier>)
- [PARegressor](<https://riverml.xyz/0.27.0/api/linear-model/PARegressor>)
- [Perceptron](<https://riverml.xyz/0.27.0/api/linear-model/Perceptron>)
- [SoftmaxRegression](<https://riverml.xyz/0.27.0/api/linear-model/SoftmaxRegression>)

### base

- [GLM](<https://riverml.xyz/0.27.0/api/linear-model/base/GLM>)

## metrics

Evaluation metrics.

All the metrics are updated one sample at a time. This way we can track performance of predictive methods over time.

Note that all metrics have a `revert` method, enabling them to be wrapped in [`utils.Rolling`](<https://riverml.xyz/0.27.0/api/utils/Rolling>). This allows computirng rolling metrics:

```py
from river import metrics, utils

y_true = [True, False, True, True]
y_pred = [False, False, True, True]

metric = utils.Rolling(metrics.Accuracy, window_size=3)

for yt, yp in zip(y_true, y_pred):
    print(metric.update(yt, yp))
```

```text
Accuracy: 0.00%
Accuracy: 50.00%
Accuracy: 66.67%
Accuracy: 100.00%
```

- [Accuracy](<https://riverml.xyz/0.27.0/api/metrics/Accuracy>)
- [AdjustedMutualInfo](<https://riverml.xyz/0.27.0/api/metrics/AdjustedMutualInfo>)
- [AdjustedRand](<https://riverml.xyz/0.27.0/api/metrics/AdjustedRand>)
- [BalancedAccuracy](<https://riverml.xyz/0.27.0/api/metrics/BalancedAccuracy>)
- [ClassificationReport](<https://riverml.xyz/0.27.0/api/metrics/ClassificationReport>)
- [CohenKappa](<https://riverml.xyz/0.27.0/api/metrics/CohenKappa>)
- [Completeness](<https://riverml.xyz/0.27.0/api/metrics/Completeness>)
- [ConfusionMatrix](<https://riverml.xyz/0.27.0/api/metrics/ConfusionMatrix>)
- [CrossEntropy](<https://riverml.xyz/0.27.0/api/metrics/CrossEntropy>)
- [F1](<https://riverml.xyz/0.27.0/api/metrics/F1>)
- [FBeta](<https://riverml.xyz/0.27.0/api/metrics/FBeta>)
- [FowlkesMallows](<https://riverml.xyz/0.27.0/api/metrics/FowlkesMallows>)
- [GeometricMean](<https://riverml.xyz/0.27.0/api/metrics/GeometricMean>)
- [Homogeneity](<https://riverml.xyz/0.27.0/api/metrics/Homogeneity>)
- [Jaccard](<https://riverml.xyz/0.27.0/api/metrics/Jaccard>)
- [LogLoss](<https://riverml.xyz/0.27.0/api/metrics/LogLoss>)
- [MAE](<https://riverml.xyz/0.27.0/api/metrics/MAE>)
- [MAPE](<https://riverml.xyz/0.27.0/api/metrics/MAPE>)
- [MCC](<https://riverml.xyz/0.27.0/api/metrics/MCC>)
- [MSE](<https://riverml.xyz/0.27.0/api/metrics/MSE>)
- [MacroF1](<https://riverml.xyz/0.27.0/api/metrics/MacroF1>)
- [MacroFBeta](<https://riverml.xyz/0.27.0/api/metrics/MacroFBeta>)
- [MacroJaccard](<https://riverml.xyz/0.27.0/api/metrics/MacroJaccard>)
- [MacroPrecision](<https://riverml.xyz/0.27.0/api/metrics/MacroPrecision>)
- [MacroRecall](<https://riverml.xyz/0.27.0/api/metrics/MacroRecall>)
- [MicroF1](<https://riverml.xyz/0.27.0/api/metrics/MicroF1>)
- [MicroFBeta](<https://riverml.xyz/0.27.0/api/metrics/MicroFBeta>)
- [MicroJaccard](<https://riverml.xyz/0.27.0/api/metrics/MicroJaccard>)
- [MicroPrecision](<https://riverml.xyz/0.27.0/api/metrics/MicroPrecision>)
- [MicroRecall](<https://riverml.xyz/0.27.0/api/metrics/MicroRecall>)
- [MultiFBeta](<https://riverml.xyz/0.27.0/api/metrics/MultiFBeta>)
- [MutualInfo](<https://riverml.xyz/0.27.0/api/metrics/MutualInfo>)
- [NormalizedMutualInfo](<https://riverml.xyz/0.27.0/api/metrics/NormalizedMutualInfo>)
- [Precision](<https://riverml.xyz/0.27.0/api/metrics/Precision>)
- [R2](<https://riverml.xyz/0.27.0/api/metrics/R2>)
- [RMSE](<https://riverml.xyz/0.27.0/api/metrics/RMSE>)
- [RMSLE](<https://riverml.xyz/0.27.0/api/metrics/RMSLE>)
- [ROCAUC](<https://riverml.xyz/0.27.0/api/metrics/ROCAUC>)
- [Rand](<https://riverml.xyz/0.27.0/api/metrics/Rand>)
- [Recall](<https://riverml.xyz/0.27.0/api/metrics/Recall>)
- [RollingPRAUC](<https://riverml.xyz/0.27.0/api/metrics/RollingPRAUC>)
- [RollingROCAUC](<https://riverml.xyz/0.27.0/api/metrics/RollingROCAUC>)
- [SMAPE](<https://riverml.xyz/0.27.0/api/metrics/SMAPE>)
- [Silhouette](<https://riverml.xyz/0.27.0/api/metrics/Silhouette>)
- [VBeta](<https://riverml.xyz/0.27.0/api/metrics/VBeta>)
- [WeightedF1](<https://riverml.xyz/0.27.0/api/metrics/WeightedF1>)
- [WeightedFBeta](<https://riverml.xyz/0.27.0/api/metrics/WeightedFBeta>)
- [WeightedJaccard](<https://riverml.xyz/0.27.0/api/metrics/WeightedJaccard>)
- [WeightedPrecision](<https://riverml.xyz/0.27.0/api/metrics/WeightedPrecision>)
- [WeightedRecall](<https://riverml.xyz/0.27.0/api/metrics/WeightedRecall>)

### base

- [BinaryMetric](<https://riverml.xyz/0.27.0/api/metrics/base/BinaryMetric>)
- [ClassificationMetric](<https://riverml.xyz/0.27.0/api/metrics/base/ClassificationMetric>)
- [Metric](<https://riverml.xyz/0.27.0/api/metrics/base/Metric>)
- [Metrics](<https://riverml.xyz/0.27.0/api/metrics/base/Metrics>)
- [MultiClassMetric](<https://riverml.xyz/0.27.0/api/metrics/base/MultiClassMetric>)
- [RegressionMetric](<https://riverml.xyz/0.27.0/api/metrics/base/RegressionMetric>)
- [WrapperMetric](<https://riverml.xyz/0.27.0/api/metrics/base/WrapperMetric>)

### multioutput

Metrics for multi-output learning.

- [ExactMatch](<https://riverml.xyz/0.27.0/api/metrics/multioutput/ExactMatch>)
- [MacroAverage](<https://riverml.xyz/0.27.0/api/metrics/multioutput/MacroAverage>)
- [MicroAverage](<https://riverml.xyz/0.27.0/api/metrics/multioutput/MicroAverage>)
- [MultiLabelConfusionMatrix](<https://riverml.xyz/0.27.0/api/metrics/multioutput/MultiLabelConfusionMatrix>)
- [PerOutput](<https://riverml.xyz/0.27.0/api/metrics/multioutput/PerOutput>)
- [SampleAverage](<https://riverml.xyz/0.27.0/api/metrics/multioutput/SampleAverage>)

#### base

- [MultiOutputClassificationMetric](<https://riverml.xyz/0.27.0/api/metrics/multioutput/base/MultiOutputClassificationMetric>)
- [MultiOutputRegressionMetric](<https://riverml.xyz/0.27.0/api/metrics/multioutput/base/MultiOutputRegressionMetric>)

## misc

Miscellaneous.

This module essentially regroups some implementations that have nowhere else to go.

- [SDFT](<https://riverml.xyz/0.27.0/api/misc/SDFT>)
- [Skyline](<https://riverml.xyz/0.27.0/api/misc/Skyline>)
- [ZstdClassifier](<https://riverml.xyz/0.27.0/api/misc/ZstdClassifier>)

## model\_selection

Model selection.

This module regroups a variety of methods that may be used for performing model selection. An model selector is provided with a list of models. These are called "experts" in the expert learning literature. The model selector's goal is to perform at least as well as the best model. Indeed, initially, the best model is not known. The performance of each model becomes more apparent as time goes by. Different strategies are possible, each one offering a different tradeoff in terms of accuracy and computational performance.

Model selection can be used for tuning the hyperparameters of a model. This may be done by creating a copy of the model for each set of hyperparameters, and treating each copy as a separate model. The [`utils.expand_param_grid`](<https://riverml.xyz/0.27.0/api/utils/expand-param-grid>) function can be used for this purpose.

- [BanditClassifier](<https://riverml.xyz/0.27.0/api/model-selection/BanditClassifier>)
- [BanditRegressor](<https://riverml.xyz/0.27.0/api/model-selection/BanditRegressor>)
- [GreedyRegressor](<https://riverml.xyz/0.27.0/api/model-selection/GreedyRegressor>)
- [HoeffdingRaceClassifier](<https://riverml.xyz/0.27.0/api/model-selection/HoeffdingRaceClassifier>)
- [HoeffdingRaceRegressor](<https://riverml.xyz/0.27.0/api/model-selection/HoeffdingRaceRegressor>)
- [SuccessiveHalvingClassifier](<https://riverml.xyz/0.27.0/api/model-selection/SuccessiveHalvingClassifier>)
- [SuccessiveHalvingRegressor](<https://riverml.xyz/0.27.0/api/model-selection/SuccessiveHalvingRegressor>)

### base

- [ModelSelectionClassifier](<https://riverml.xyz/0.27.0/api/model-selection/base/ModelSelectionClassifier>)
- [ModelSelectionRegressor](<https://riverml.xyz/0.27.0/api/model-selection/base/ModelSelectionRegressor>)

## multiclass

Multi-class classification.

- [OneVsOneClassifier](<https://riverml.xyz/0.27.0/api/multiclass/OneVsOneClassifier>)
- [OneVsRestClassifier](<https://riverml.xyz/0.27.0/api/multiclass/OneVsRestClassifier>)
- [OutputCodeClassifier](<https://riverml.xyz/0.27.0/api/multiclass/OutputCodeClassifier>)

## multioutput

Multi-output models.

- [ClassifierChain](<https://riverml.xyz/0.27.0/api/multioutput/ClassifierChain>)
- [MonteCarloClassifierChain](<https://riverml.xyz/0.27.0/api/multioutput/MonteCarloClassifierChain>)
- [MultiClassEncoder](<https://riverml.xyz/0.27.0/api/multioutput/MultiClassEncoder>)
- [PerOutputClassifier](<https://riverml.xyz/0.27.0/api/multioutput/PerOutputClassifier>)
- [PerOutputRegressor](<https://riverml.xyz/0.27.0/api/multioutput/PerOutputRegressor>)
- [ProbabilisticClassifierChain](<https://riverml.xyz/0.27.0/api/multioutput/ProbabilisticClassifierChain>)
- [RegressorChain](<https://riverml.xyz/0.27.0/api/multioutput/RegressorChain>)

## naive\_bayes

Naive Bayes algorithms.

- [BernoulliNB](<https://riverml.xyz/0.27.0/api/naive-bayes/BernoulliNB>)
- [CategoricalNB](<https://riverml.xyz/0.27.0/api/naive-bayes/CategoricalNB>)
- [ComplementNB](<https://riverml.xyz/0.27.0/api/naive-bayes/ComplementNB>)
- [GaussianNB](<https://riverml.xyz/0.27.0/api/naive-bayes/GaussianNB>)
- [MultinomialNB](<https://riverml.xyz/0.27.0/api/naive-bayes/MultinomialNB>)

## neighbors

Neighbors-based learning.

Also known as *lazy* methods. In these methods, generalisation of the training data is delayed until a query is received.

- [KNNClassifier](<https://riverml.xyz/0.27.0/api/neighbors/KNNClassifier>)
- [KNNRegressor](<https://riverml.xyz/0.27.0/api/neighbors/KNNRegressor>)
- [LazySearch](<https://riverml.xyz/0.27.0/api/neighbors/LazySearch>)
- [SWINN](<https://riverml.xyz/0.27.0/api/neighbors/SWINN>)

## optim

Stochastic optimization.

- [AMSGrad](<https://riverml.xyz/0.27.0/api/optim/AMSGrad>)
- [AdaBound](<https://riverml.xyz/0.27.0/api/optim/AdaBound>)
- [AdaDelta](<https://riverml.xyz/0.27.0/api/optim/AdaDelta>)
- [AdaGrad](<https://riverml.xyz/0.27.0/api/optim/AdaGrad>)
- [AdaMax](<https://riverml.xyz/0.27.0/api/optim/AdaMax>)
- [Adam](<https://riverml.xyz/0.27.0/api/optim/Adam>)
- [Averager](<https://riverml.xyz/0.27.0/api/optim/Averager>)
- [FTRLProximal](<https://riverml.xyz/0.27.0/api/optim/FTRLProximal>)
- [Momentum](<https://riverml.xyz/0.27.0/api/optim/Momentum>)
- [Nadam](<https://riverml.xyz/0.27.0/api/optim/Nadam>)
- [NesterovMomentum](<https://riverml.xyz/0.27.0/api/optim/NesterovMomentum>)
- [Newton](<https://riverml.xyz/0.27.0/api/optim/Newton>)
- [Optimizer](<https://riverml.xyz/0.27.0/api/optim/Optimizer>)
- [RMSProp](<https://riverml.xyz/0.27.0/api/optim/RMSProp>)
- [SGD](<https://riverml.xyz/0.27.0/api/optim/SGD>)

### base

- [Initializer](<https://riverml.xyz/0.27.0/api/optim/base/Initializer>)
- [Loss](<https://riverml.xyz/0.27.0/api/optim/base/Loss>)
- [Optimizer](<https://riverml.xyz/0.27.0/api/optim/base/Optimizer>)
- [Scheduler](<https://riverml.xyz/0.27.0/api/optim/base/Scheduler>)

### initializers

Weight initializers.

- [Constant](<https://riverml.xyz/0.27.0/api/optim/initializers/Constant>)
- [Normal](<https://riverml.xyz/0.27.0/api/optim/initializers/Normal>)
- [Zeros](<https://riverml.xyz/0.27.0/api/optim/initializers/Zeros>)

### losses

Loss functions.

Each loss function is intended to work with both single values as well as numpy vectors.

## Boundary convention

Several losses are piecewise (`Hinge`, `Absolute`, `EpsilonInsensitiveHinge`, `Huber`, ...), so their gradient is discontinuous at a kink. The invariant we hold everywhere is that the scalar and vectorised branches of a loss **must** resolve such a boundary point identically; otherwise looping `learn_one` and a single `learn_many` call would disagree on inputs that land exactly on the kink.

- [Absolute](<https://riverml.xyz/0.27.0/api/optim/losses/Absolute>)
- [BinaryFocalLoss](<https://riverml.xyz/0.27.0/api/optim/losses/BinaryFocalLoss>)
- [BinaryLoss](<https://riverml.xyz/0.27.0/api/optim/losses/BinaryLoss>)
- [Cauchy](<https://riverml.xyz/0.27.0/api/optim/losses/Cauchy>)
- [CrossEntropy](<https://riverml.xyz/0.27.0/api/optim/losses/CrossEntropy>)
- [EpsilonInsensitiveHinge](<https://riverml.xyz/0.27.0/api/optim/losses/EpsilonInsensitiveHinge>)
- [Hinge](<https://riverml.xyz/0.27.0/api/optim/losses/Hinge>)
- [Huber](<https://riverml.xyz/0.27.0/api/optim/losses/Huber>)
- [Log](<https://riverml.xyz/0.27.0/api/optim/losses/Log>)
- [MultiClassLoss](<https://riverml.xyz/0.27.0/api/optim/losses/MultiClassLoss>)
- [Poisson](<https://riverml.xyz/0.27.0/api/optim/losses/Poisson>)
- [Quantile](<https://riverml.xyz/0.27.0/api/optim/losses/Quantile>)
- [RegressionLoss](<https://riverml.xyz/0.27.0/api/optim/losses/RegressionLoss>)
- [Squared](<https://riverml.xyz/0.27.0/api/optim/losses/Squared>)

### schedulers

Learning rate schedulers.

- [Constant](<https://riverml.xyz/0.27.0/api/optim/schedulers/Constant>)
- [InverseScaling](<https://riverml.xyz/0.27.0/api/optim/schedulers/InverseScaling>)
- [Optimal](<https://riverml.xyz/0.27.0/api/optim/schedulers/Optimal>)

## preprocessing

Feature preprocessing.

The purpose of this module is to modify an existing set of features so that they can be processed by a machine learning algorithm. This may be done by scaling numeric parts of the data or by one-hot encoding categorical features. The difference with the `feature_extraction` module is that the latter extracts new information from the data

- [AdaptiveStandardScaler](<https://riverml.xyz/0.27.0/api/preprocessing/AdaptiveStandardScaler>)
- [Binarizer](<https://riverml.xyz/0.27.0/api/preprocessing/Binarizer>)
- [FeatureHasher](<https://riverml.xyz/0.27.0/api/preprocessing/FeatureHasher>)
- [GapEncoder](<https://riverml.xyz/0.27.0/api/preprocessing/GapEncoder>)
- [GaussianRandomProjector](<https://riverml.xyz/0.27.0/api/preprocessing/GaussianRandomProjector>)
- [LDA](<https://riverml.xyz/0.27.0/api/preprocessing/LDA>)
- [MaxAbsScaler](<https://riverml.xyz/0.27.0/api/preprocessing/MaxAbsScaler>)
- [MinMaxScaler](<https://riverml.xyz/0.27.0/api/preprocessing/MinMaxScaler>)
- [Normalizer](<https://riverml.xyz/0.27.0/api/preprocessing/Normalizer>)
- [OneHotEncoder](<https://riverml.xyz/0.27.0/api/preprocessing/OneHotEncoder>)
- [OrdinalEncoder](<https://riverml.xyz/0.27.0/api/preprocessing/OrdinalEncoder>)
- [PredClipper](<https://riverml.xyz/0.27.0/api/preprocessing/PredClipper>)
- [PreviousImputer](<https://riverml.xyz/0.27.0/api/preprocessing/PreviousImputer>)
- [RobustScaler](<https://riverml.xyz/0.27.0/api/preprocessing/RobustScaler>)
- [SparseRandomProjector](<https://riverml.xyz/0.27.0/api/preprocessing/SparseRandomProjector>)
- [StandardScaler](<https://riverml.xyz/0.27.0/api/preprocessing/StandardScaler>)
- [StatImputer](<https://riverml.xyz/0.27.0/api/preprocessing/StatImputer>)
- [TargetMinMaxScaler](<https://riverml.xyz/0.27.0/api/preprocessing/TargetMinMaxScaler>)
- [TargetStandardScaler](<https://riverml.xyz/0.27.0/api/preprocessing/TargetStandardScaler>)

## proba

Probability distributions.

- [Beta](<https://riverml.xyz/0.27.0/api/proba/Beta>)
- [Gaussian](<https://riverml.xyz/0.27.0/api/proba/Gaussian>)
- [Multinomial](<https://riverml.xyz/0.27.0/api/proba/Multinomial>)
- [MultivariateGaussian](<https://riverml.xyz/0.27.0/api/proba/MultivariateGaussian>)

### base

- [BinaryDistribution](<https://riverml.xyz/0.27.0/api/proba/base/BinaryDistribution>)
- [ContinuousDistribution](<https://riverml.xyz/0.27.0/api/proba/base/ContinuousDistribution>)
- [DiscreteDistribution](<https://riverml.xyz/0.27.0/api/proba/base/DiscreteDistribution>)
- [Distribution](<https://riverml.xyz/0.27.0/api/proba/base/Distribution>)

## reco

Recommender systems module.

Recommender systems (recsys for short) is a large topic. This module is far from comprehensive. It simply provides models which can contribute towards building a recommender system.

A typical recommender system is made up of a retrieval phase, followed by a ranking phase. The output of the retrieval phase is a shortlist of the catalogue of items. The items in the shortlist are then usually ranked according to the expected preference the user will have for each item. This module focuses on the ranking phase.

Models which inherit from the `Ranker` class have a `rank` method. This allows sorting a set of items for a given user. Each model also has a `learn_one(user, item, y, context)` which allows learning user preferences. The `y` parameter is a reward value, the nature of which depends is specific to each and every recommendation task. Typically the reward is a number or a boolean value. It is up to the user to determine how to translate a user session into training data.

- [Baseline](<https://riverml.xyz/0.27.0/api/reco/Baseline>)
- [BiasedMF](<https://riverml.xyz/0.27.0/api/reco/BiasedMF>)
- [FunkMF](<https://riverml.xyz/0.27.0/api/reco/FunkMF>)
- [RandomNormal](<https://riverml.xyz/0.27.0/api/reco/RandomNormal>)

### base

- [Ranker](<https://riverml.xyz/0.27.0/api/reco/base/Ranker>)

## rules

Decision rules-based algorithms.

- [AMRules](<https://riverml.xyz/0.27.0/api/rules/AMRules>)

## sketch

Data containers and collections for sequential data.

This module has summary and sketch structures that operate with constrained amounts of memory and processing time.

- [Counter](<https://riverml.xyz/0.27.0/api/sketch/Counter>)
- [HeavyHitters](<https://riverml.xyz/0.27.0/api/sketch/HeavyHitters>)
- [Histogram](<https://riverml.xyz/0.27.0/api/sketch/Histogram>)
- [NUnique](<https://riverml.xyz/0.27.0/api/sketch/NUnique>)
- [Set](<https://riverml.xyz/0.27.0/api/sketch/Set>)

## stats

Running statistics.

- [AbsMax](<https://riverml.xyz/0.27.0/api/stats/AbsMax>)
- [AutoCorr](<https://riverml.xyz/0.27.0/api/stats/AutoCorr>)
- [BayesianMean](<https://riverml.xyz/0.27.0/api/stats/BayesianMean>)
- [ChiSquared](<https://riverml.xyz/0.27.0/api/stats/ChiSquared>)
- [Count](<https://riverml.xyz/0.27.0/api/stats/Count>)
- [Cov](<https://riverml.xyz/0.27.0/api/stats/Cov>)
- [EWCov](<https://riverml.xyz/0.27.0/api/stats/EWCov>)
- [EWMean](<https://riverml.xyz/0.27.0/api/stats/EWMean>)
- [EWVar](<https://riverml.xyz/0.27.0/api/stats/EWVar>)
- [Entropy](<https://riverml.xyz/0.27.0/api/stats/Entropy>)
- [IQR](<https://riverml.xyz/0.27.0/api/stats/IQR>)
- [KolmogorovSmirnov](<https://riverml.xyz/0.27.0/api/stats/KolmogorovSmirnov>)
- [Kurtosis](<https://riverml.xyz/0.27.0/api/stats/Kurtosis>)
- [Link](<https://riverml.xyz/0.27.0/api/stats/Link>)
- [MAD](<https://riverml.xyz/0.27.0/api/stats/MAD>)
- [Max](<https://riverml.xyz/0.27.0/api/stats/Max>)
- [Mean](<https://riverml.xyz/0.27.0/api/stats/Mean>)
- [Min](<https://riverml.xyz/0.27.0/api/stats/Min>)
- [Mode](<https://riverml.xyz/0.27.0/api/stats/Mode>)
- [PeakToPeak](<https://riverml.xyz/0.27.0/api/stats/PeakToPeak>)
- [PearsonCorr](<https://riverml.xyz/0.27.0/api/stats/PearsonCorr>)
- [Quantile](<https://riverml.xyz/0.27.0/api/stats/Quantile>)
- [RollingAbsMax](<https://riverml.xyz/0.27.0/api/stats/RollingAbsMax>)
- [RollingIQR](<https://riverml.xyz/0.27.0/api/stats/RollingIQR>)
- [RollingMax](<https://riverml.xyz/0.27.0/api/stats/RollingMax>)
- [RollingMin](<https://riverml.xyz/0.27.0/api/stats/RollingMin>)
- [RollingMode](<https://riverml.xyz/0.27.0/api/stats/RollingMode>)
- [RollingPeakToPeak](<https://riverml.xyz/0.27.0/api/stats/RollingPeakToPeak>)
- [RollingQuantile](<https://riverml.xyz/0.27.0/api/stats/RollingQuantile>)
- [SEM](<https://riverml.xyz/0.27.0/api/stats/SEM>)
- [Shift](<https://riverml.xyz/0.27.0/api/stats/Shift>)
- [Skew](<https://riverml.xyz/0.27.0/api/stats/Skew>)
- [Sum](<https://riverml.xyz/0.27.0/api/stats/Sum>)
- [Var](<https://riverml.xyz/0.27.0/api/stats/Var>)

### base

- [Bivariate](<https://riverml.xyz/0.27.0/api/stats/base/Bivariate>)
- [Link](<https://riverml.xyz/0.27.0/api/stats/base/Link>)
- [RollingUnivariate](<https://riverml.xyz/0.27.0/api/stats/base/RollingUnivariate>)
- [Statistic](<https://riverml.xyz/0.27.0/api/stats/base/Statistic>)
- [Univariate](<https://riverml.xyz/0.27.0/api/stats/base/Univariate>)

## stream

Streaming utilities.

The module includes tools to iterate over data streams.

**Classes**

- [Cache](<https://riverml.xyz/0.27.0/api/stream/Cache>)
- [TwitchChatStream](<https://riverml.xyz/0.27.0/api/stream/TwitchChatStream>)
- [TwitterLiveStream](<https://riverml.xyz/0.27.0/api/stream/TwitterLiveStream>)

**Functions**

- [iter\_arff](<https://riverml.xyz/0.27.0/api/stream/iter-arff>)
- [iter\_array](<https://riverml.xyz/0.27.0/api/stream/iter-array>)
- [iter\_csv](<https://riverml.xyz/0.27.0/api/stream/iter-csv>)
- [iter\_frame](<https://riverml.xyz/0.27.0/api/stream/iter-frame>)
- [iter\_libsvm](<https://riverml.xyz/0.27.0/api/stream/iter-libsvm>)
- [iter\_pandas](<https://riverml.xyz/0.27.0/api/stream/iter-pandas>)
- [iter\_polars](<https://riverml.xyz/0.27.0/api/stream/iter-polars>)
- [iter\_sklearn\_dataset](<https://riverml.xyz/0.27.0/api/stream/iter-sklearn-dataset>)
- [iter\_sql](<https://riverml.xyz/0.27.0/api/stream/iter-sql>)
- [shuffle](<https://riverml.xyz/0.27.0/api/stream/shuffle>)
- [simulate\_qa](<https://riverml.xyz/0.27.0/api/stream/simulate-qa>)

## time\_series

Time series forecasting.

**Classes**

- [ForecastingMetric](<https://riverml.xyz/0.27.0/api/time-series/ForecastingMetric>)
- [HoltWinters](<https://riverml.xyz/0.27.0/api/time-series/HoltWinters>)
- [HorizonAggMetric](<https://riverml.xyz/0.27.0/api/time-series/HorizonAggMetric>)
- [HorizonMetric](<https://riverml.xyz/0.27.0/api/time-series/HorizonMetric>)
- [SNARIMAX](<https://riverml.xyz/0.27.0/api/time-series/SNARIMAX>)

**Functions**

- [evaluate](<https://riverml.xyz/0.27.0/api/time-series/evaluate>)
- [iter\_evaluate](<https://riverml.xyz/0.27.0/api/time-series/iter-evaluate>)

### base

- [Forecaster](<https://riverml.xyz/0.27.0/api/time-series/base/Forecaster>)

## tree

Incremental decision trees.

This module implements incremental Decision Tree (iDT) algorithms for handling classification and regression tasks.

Each family of iDT will be presented in a dedicated section.

At any moment, iDT might face situations where an input feature previously used to make a split decision is missing in an incoming sample. In this case, the most traversed path is selected to pass down the instance. Moreover, in the case of nominal features, if a new category arises and the feature is used in a decision node, a new branch is created to accommodate the new value.

**1\. Hoeffding Trees**

This family of iDT algorithms use the Hoeffding Bound to determine whether or not the incrementally computed best split candidates would be equivalent to the ones obtained in a batch-processing fashion.

All the available Hoeffding Tree (HT) implementation share some common functionalities:

- Set the maximum tree depth allowed (`max_depth`).
- Handle *Active* and *Inactive* nodes: Active learning nodes update their own internal state to improve predictions and monitor input features to perform split attempts. Inactive learning nodes do not update their internal state and only keep the predictors; they are used to save memory in the tree (`max_size`).
- Enable/disable memory management.
- Define strategies to sort leaves according to how likely they are going to be split. This enables deactivating non-promising leaves to save memory.
- Disabling ‘poor’ attributes to save memory and speed up tree construction. A poor attribute is an input feature whose split merit is much smaller than the current best candidate. Once a feature is disabled, the tree stops saving statistics necessary to split such a feature.
- Define properties to access leaf prediction strategies, split criteria, and other relevant characteristics.

**2\. Stochastic Gradient Trees**

Stochastic Gradient Trees (SGT) directly optimize a loss function, rather than relying on split heuristics to guide the tree growth. F-tests are performed do decide whether a leaf should be expanded or its prediction value should be updated.

SGTs can deal with binary classification and single-target regression. They also support dynamic and static feature quantizers to deal with numerical inputs.

- [ExtremelyFastDecisionTreeClassifier](<https://riverml.xyz/0.27.0/api/tree/ExtremelyFastDecisionTreeClassifier>)
- [HoeffdingAdaptiveTreeClassifier](<https://riverml.xyz/0.27.0/api/tree/HoeffdingAdaptiveTreeClassifier>)
- [HoeffdingAdaptiveTreeRegressor](<https://riverml.xyz/0.27.0/api/tree/HoeffdingAdaptiveTreeRegressor>)
- [HoeffdingTreeClassifier](<https://riverml.xyz/0.27.0/api/tree/HoeffdingTreeClassifier>)
- [HoeffdingTreeRegressor](<https://riverml.xyz/0.27.0/api/tree/HoeffdingTreeRegressor>)
- [ISOUPTreeRegressor](<https://riverml.xyz/0.27.0/api/tree/ISOUPTreeRegressor>)
- [LASTClassifier](<https://riverml.xyz/0.27.0/api/tree/LASTClassifier>)
- [SGTClassifier](<https://riverml.xyz/0.27.0/api/tree/SGTClassifier>)
- [SGTRegressor](<https://riverml.xyz/0.27.0/api/tree/SGTRegressor>)

### base

Generic branches and leaves for decision trees.

This module defines generic branch and leaf implementations. These should be used in River by each tree-based model. Using these classes makes the code more DRY. The only exception for not doing so would be for performance, whereby a tree-based model uses a bespoke implementation.

This module defines a bunch of methods to ease the manipulation and diagnostic of trees. Its intention is to provide utilities for walking over a tree and visualizing it.

- [Branch](<https://riverml.xyz/0.27.0/api/tree/base/Branch>)
- [Leaf](<https://riverml.xyz/0.27.0/api/tree/base/Leaf>)

### splitter

Leaf splitters for decision trees.

This module implements the Attribute Observers (AO) (or tree splitters) that are used by the Hoeffding Trees (HT). It also implements the feature quantizers (FQ) used by Stochastic Gradient Trees (SGT). AOs are a core aspect of the HTs construction, and might represent one of the major bottlenecks when building the trees. The same holds for SGTs and FQs. The correct choice and setup of a splitter might result in significant differences in the running time and memory usage of the incremental decision trees.

AOs for classification and regression trees can be differentiated by using the property `is_target_class` (`True` for splitters designed to classification tasks). An error will be raised if one tries to use a classification splitter in a regression tree and vice-versa. Lastly, AOs cannot be used in SGT and FQs cannot be used in Hoeffding Trees. So, care must be taken when choosing the correct feature splitter.

- [DynamicQuantizer](<https://riverml.xyz/0.27.0/api/tree/splitter/DynamicQuantizer>)
- [EBSTSplitter](<https://riverml.xyz/0.27.0/api/tree/splitter/EBSTSplitter>)
- [ExhaustiveSplitter](<https://riverml.xyz/0.27.0/api/tree/splitter/ExhaustiveSplitter>)
- [GaussianSplitter](<https://riverml.xyz/0.27.0/api/tree/splitter/GaussianSplitter>)
- [HistogramSplitter](<https://riverml.xyz/0.27.0/api/tree/splitter/HistogramSplitter>)
- [QOSplitter](<https://riverml.xyz/0.27.0/api/tree/splitter/QOSplitter>)
- [Quantizer](<https://riverml.xyz/0.27.0/api/tree/splitter/Quantizer>)
- [Splitter](<https://riverml.xyz/0.27.0/api/tree/splitter/Splitter>)
- [StaticQuantizer](<https://riverml.xyz/0.27.0/api/tree/splitter/StaticQuantizer>)
- [TEBSTSplitter](<https://riverml.xyz/0.27.0/api/tree/splitter/TEBSTSplitter>)

## utils

Shared utility classes and functions.

**Classes**

- [Rolling](<https://riverml.xyz/0.27.0/api/utils/Rolling>)
- [SortedWindow](<https://riverml.xyz/0.27.0/api/utils/SortedWindow>)
- [TimeRolling](<https://riverml.xyz/0.27.0/api/utils/TimeRolling>)
- [VectorDict](<https://riverml.xyz/0.27.0/api/utils/VectorDict>)

**Functions**

- [expand\_param\_grid](<https://riverml.xyz/0.27.0/api/utils/expand-param-grid>)
- [log\_method\_calls](<https://riverml.xyz/0.27.0/api/utils/log-method-calls>)

### math

Mathematical utility functions (intended for internal purposes).

A lot of this is experimental and has a high probability of changing in the future.

- [argmax](<https://riverml.xyz/0.27.0/api/utils/math/argmax>)
- [chain\_dot](<https://riverml.xyz/0.27.0/api/utils/math/chain-dot>)
- [clamp](<https://riverml.xyz/0.27.0/api/utils/math/clamp>)
- [dot](<https://riverml.xyz/0.27.0/api/utils/math/dot>)
- [dotvecmat](<https://riverml.xyz/0.27.0/api/utils/math/dotvecmat>)
- [log\_sum\_2\_exp](<https://riverml.xyz/0.27.0/api/utils/math/log-sum-2-exp>)
- [matmul2d](<https://riverml.xyz/0.27.0/api/utils/math/matmul2d>)
- [minkowski\_distance](<https://riverml.xyz/0.27.0/api/utils/math/minkowski-distance>)
- [norm](<https://riverml.xyz/0.27.0/api/utils/math/norm>)
- [outer](<https://riverml.xyz/0.27.0/api/utils/math/outer>)
- [prod](<https://riverml.xyz/0.27.0/api/utils/math/prod>)
- [sherman\_morrison](<https://riverml.xyz/0.27.0/api/utils/math/sherman-morrison>)
- [sigmoid](<https://riverml.xyz/0.27.0/api/utils/math/sigmoid>)
- [sign](<https://riverml.xyz/0.27.0/api/utils/math/sign>)
- [softmax](<https://riverml.xyz/0.27.0/api/utils/math/softmax>)
- [woodbury\_matrix](<https://riverml.xyz/0.27.0/api/utils/math/woodbury-matrix>)

### norm

- [normalize\_values\_in\_dict](<https://riverml.xyz/0.27.0/api/utils/norm/normalize-values-in-dict>)
- [scale\_values\_in\_dict](<https://riverml.xyz/0.27.0/api/utils/norm/scale-values-in-dict>)

### pretty

Helper functions for making things readable by humans.

- [humanize\_bytes](<https://riverml.xyz/0.27.0/api/utils/pretty/humanize-bytes>)
- [print\_table](<https://riverml.xyz/0.27.0/api/utils/pretty/print-table>)

### random

- [exponential](<https://riverml.xyz/0.27.0/api/utils/random/exponential>)
- [poisson](<https://riverml.xyz/0.27.0/api/utils/random/poisson>)
