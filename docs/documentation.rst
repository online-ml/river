=================
API Documentation
=================

This is the API documentation for ``scikit-multiflow``.

.. _data_ref:

Core: :mod:`skmultiflow.core`
=============================

.. automodule:: skmultiflow.core
    :no-members:
    :no-inherited-members:

.. currentmodule:: skmultiflow

.. autosummary::
   :nosignatures:
   :toctree: _autosummary

   core.base.BaseEstimator
   core.BaseSKMObject
   core.ClassifierMixin
   core.RegressorMixin
   core.MetaEstimatorMixin
   core.MultiOutputMixin
   core.Pipeline

Data: :mod:`skmultiflow.data`
=============================

.. automodule:: skmultiflow.data
    :no-members:
    :no-inherited-members:

.. currentmodule:: skmultiflow

.. autosummary::
   :nosignatures:
   :toctree: _autosummary

   data.base_stream.Stream
   data.DataStream
   data.FileStream

Stream Generators
-----------------

.. currentmodule:: skmultiflow

.. autosummary::
   :nosignatures:
   :toctree: _autosummary

   data.AGRAWALGenerator
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
   data.ConceptDriftStream

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
   :toctree: _autosummary

   anomaly_detection.HalfSpaceTrees

Bayes methods
-------------

.. automodule:: skmultiflow.bayes
    :no-members:
    :no-inherited-members:

.. currentmodule:: skmultiflow

.. autosummary::
   :nosignatures:
   :toctree: _autosummary

   bayes.NaiveBayes

Lazy learning methods
---------------------

.. automodule:: skmultiflow.lazy
   :no-members:
   :no-inherited-members:

.. currentmodule:: skmultiflow

.. autosummary::
   :nosignatures:
   :toctree: _autosummary

   lazy.KNN
   lazy.KNNAdwin
   lazy.SAMKNN

Ensemble methods
----------------

.. automodule:: skmultiflow.meta
   :no-members:
   :no-inherited-members:

.. currentmodule:: skmultiflow

.. autosummary::
   :nosignatures:
   :toctree: _autosummary

   meta.AccuracyWeightedEnsemble
   meta.AdaptiveRandomForest
   meta.AdditiveExpertEnsemble
   meta.BatchIncremental
   meta.ClassifierChain
   meta.ProbabilisticClassifierChain
   meta.MonteCarloClassifierChain
   meta.DynamicWeightedMajority
   meta.LearnNSE
   meta.LearnPP
   meta.LeverageBagging
   meta.MultiOutputLearner
   meta.OnlineAdaC2
   meta.OnlineBoosting
   meta.OnlineCSB2
   meta.OnlineRUSBoost
   meta.OnlineSMOTEBagging
   meta.OnlineUnderOverBagging
   meta.OzaBagging
   meta.OzaBaggingAdwin
   meta.RegressorChain

Neural Networks
---------------

.. automodule:: skmultiflow.neural_networks
    :no-members:
    :no-inherited-members:

.. currentmodule:: skmultiflow

.. autosummary::
   :nosignatures:
   :toctree: _autosummary

   neural_networks.PerceptronMask

Prototype based methods
-----------------------

.. automodule:: skmultiflow.prototype
   :no-members:
   :no-inherited-members:

.. currentmodule:: skmultiflow

.. autosummary::
   :nosignatures:
   :toctree: _autosummary

       prototype.RobustSoftLearningVectorQuantization

Rules based methods
-------------------

.. automodule:: skmultiflow.rules
   :no-members:
   :no-inherited-members:

.. currentmodule:: skmultiflow

.. autosummary::
   :nosignatures:
   :toctree: _autosummary

   rules.VFDR

Trees based methods
-------------------

.. automodule:: skmultiflow.trees
   :no-members:
   :no-inherited-members:

.. currentmodule:: skmultiflow

.. autosummary::
   :nosignatures:
   :toctree: _autosummary

   trees.HoeffdingTree
   trees.HAT
   trees.HATT
   trees.LCHT
   trees.RegressionHoeffdingTree
   trees.RegressionHAT
   trees.MultiTargetRegressionHoeffdingTree
   trees.StackedSingleTargetHoeffdingTreeRegressor


Drift Detection: :mod:`skmultiflow.drift_detection`
===================================================

.. automodule:: skmultiflow.drift_detection
   :no-members:
   :no-inherited-members:

.. currentmodule:: skmultiflow

.. autosummary::
   :nosignatures:
   :toctree: _autosummary

   drift_detection.ADWIN
   drift_detection.DDM
   drift_detection.EDDM
   drift_detection.PageHinkley

Evaluation: :mod:`skmultiflow.evaluation`
=========================================

.. automodule:: skmultiflow.evaluation
   :no-members:
   :no-inherited-members:

.. currentmodule:: skmultiflow

.. autosummary::
   :nosignatures:
   :toctree: _autosummary

   evaluation.EvaluateHoldout
   evaluation.EvaluatePrequential

Transform: :mod:`skmultiflow.transform`
=======================================

.. automodule:: skmultiflow.transform
   :no-members:
   :no-inherited-members:

.. currentmodule:: skmultiflow

.. autosummary::
   :nosignatures:
   :toctree: _autosummary

   transform.MissingValuesCleaner
   transform.OneHotToCategorical

Misc:
=====

Data structure
--------------

.. currentmodule:: skmultiflow

.. autosummary::
   :nosignatures:
   :toctree: _autosummary

   lazy.KDTree

Utilities
---------

.. currentmodule:: skmultiflow

.. autosummary::
   :nosignatures:
   :toctree: _autosummary

   core.clone
