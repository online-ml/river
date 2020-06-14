=================
API Reference
=================

This is the API documentation for ``scikit-multiflow``.

.. _data_ref:

Core
====

.. automodule:: skmultiflow.core
    :no-members:
    :no-inherited-members:

.. currentmodule:: skmultiflow

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: generated

   core.base.BaseEstimator
   core.BaseSKMObject
   core.ClassifierMixin
   core.RegressorMixin
   core.MetaEstimatorMixin
   core.MultiOutputMixin
   core.Pipeline

Data
====

.. automodule:: skmultiflow.data
    :no-members:
    :no-inherited-members:

.. currentmodule:: skmultiflow

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: generated

   data.base_stream.Stream
   data.DataStream
   data.FileStream
   data.ConceptDriftStream
   data.TemporalDataStream

Stream Generators
-----------------

.. currentmodule:: skmultiflow

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: generated

   data.AGRAWALGenerator
   data.AnomalySineGenerator
   data.HyperplaneGenerator
   data.LEDGenerator
   data.LEDGeneratorDrift
   data.MIXEDGenerator
   data.RandomRBFGenerator
   data.RandomRBFGeneratorDrift
   data.RandomTreeGenerator
   data.SEAGenerator
   data.SineGenerator
   data.STAGGERGenerator
   data.WaveformGenerator
   data.MultilabelGenerator
   data.RegressionGenerator

Learning methods
================

Anomaly detection methods
-------------------------

.. automodule:: skmultiflow.anomaly_detection
    :no-members:
    :no-inherited-members:

.. currentmodule:: skmultiflow

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: generated

   anomaly_detection.HalfSpaceTrees

Bayes methods
-------------

.. automodule:: skmultiflow.bayes
    :no-members:
    :no-inherited-members:

.. currentmodule:: skmultiflow

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: generated

   bayes.NaiveBayes

Lazy learning methods
---------------------

.. automodule:: skmultiflow.lazy
   :no-members:
   :no-inherited-members:

.. currentmodule:: skmultiflow

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: generated

   lazy.KNNClassifier
   lazy.KNNADWINClassifier
   lazy.SAMKNNClassifier
   lazy.KNNRegressor

Ensemble methods
----------------

.. automodule:: skmultiflow.meta
   :no-members:
   :no-inherited-members:

.. currentmodule:: skmultiflow

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: generated

   meta.AccuracyWeightedEnsembleClassifier
   meta.AdaptiveRandomForestClassifier
   meta.AdaptiveRandomForestRegressor
   meta.AdditiveExpertEnsembleClassifier
   meta.BatchIncrementalClassifier
   meta.ClassifierChain
   meta.ProbabilisticClassifierChain
   meta.MonteCarloClassifierChain
   meta.DynamicWeightedMajorityClassifier
   meta.LearnPPNSEClassifier
   meta.LearnPPClassifier
   meta.LeveragingBaggingClassifier
   meta.MultiOutputLearner
   meta.OnlineAdaC2Classifier
   meta.OnlineBoostingClassifier
   meta.OnlineCSB2Classifier
   meta.OnlineRUSBoostClassifier
   meta.OnlineSMOTEBaggingClassifier
   meta.OnlineUnderOverBaggingClassifier
   meta.OzaBaggingClassifier
   meta.OzaBaggingADWINClassifier
   meta.RegressorChain
   meta.StreamingRandomPatchesClassifier

Neural Networks
---------------

.. automodule:: skmultiflow.neural_networks
    :no-members:
    :no-inherited-members:

.. currentmodule:: skmultiflow

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: generated

   neural_networks.PerceptronMask

Prototype based methods
-----------------------

.. automodule:: skmultiflow.prototype
   :no-members:
   :no-inherited-members:

.. currentmodule:: skmultiflow

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: generated

       prototype.RobustSoftLearningVectorQuantization

Rules based methods
-------------------

.. automodule:: skmultiflow.rules
   :no-members:
   :no-inherited-members:

.. currentmodule:: skmultiflow

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: generated

   rules.VeryFastDecisionRulesClassifier

Trees based methods
-------------------

.. automodule:: skmultiflow.trees
   :no-members:
   :no-inherited-members:

.. currentmodule:: skmultiflow

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: generated

   trees.HoeffdingTreeClassifier
   trees.HoeffdingAdaptiveTreeClassifier
   trees.ExtremelyFastDecisionTreeClassifier
   trees.LabelCombinationHoeffdingTreeClassifier
   trees.HoeffdingTreeRegressor
   trees.HoeffdingAdaptiveTreeRegressor
   trees.iSOUPTreeRegressor
   trees.StackedSingleTargetHoeffdingTreeRegressor


Drift Detection
===============

.. automodule:: skmultiflow.drift_detection
   :no-members:
   :no-inherited-members:

.. currentmodule:: skmultiflow

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: generated

   drift_detection.ADWIN
   drift_detection.DDM
   drift_detection.EDDM
   drift_detection.HDDM_A
   drift_detection.HDDM_W
   drift_detection.KSWIN
   drift_detection.PageHinkley

Evaluation
==========

.. automodule:: skmultiflow.evaluation
   :no-members:
   :no-inherited-members:

.. currentmodule:: skmultiflow

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: generated

   evaluation.EvaluateHoldout
   evaluation.EvaluatePrequential
   evaluation.EvaluatePrequentialDelayed

Transform
=========

.. automodule:: skmultiflow.transform
   :no-members:
   :no-inherited-members:

.. currentmodule:: skmultiflow

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: generated

   transform.MissingValuesCleaner
   transform.OneHotToCategorical
   transform.WindowedMinmaxScaler
   transform.WindowedStandardScaler

Misc
====

Utilities
---------

.. currentmodule:: skmultiflow

.. autosummary::
   :nosignatures:
   :toctree: generated

   core.clone
