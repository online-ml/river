=================
API Documentation
=================

This is the API documentation for ``scikit-multiflow``.

.. _data_ref:

Data: :mod:`skmultiflow.data`
=============================

.. automodule:: skmultiflow.data
    :no-members:
    :no-inherited-members:

.. currentmodule:: skmultiflow

.. autosummary::
   :toctree: _autosummary

   data.base_stream
   data.data_stream
   data.file_stream

Stream Generators
-----------------

.. currentmodule:: skmultiflow

.. autosummary::
   :toctree: _autosummary

   data.agrawal_generator
   data.hyper_plane_generator
   data.led_generator
   data.led_generator_drift
   data.mixed_generator
   data.random_rbf_generator
   data.random_rbf_generator_drift
   data.random_tree_generator
   data.sea_generator
   data.sine_generator
   data.stagger_generator
   data.waveform_generator
   data.multilabel_generator
   data.regression_generator
   data.concept_drift_stream

Learning methods
================

Bayes methods
-------------

.. automodule:: skmultiflow.bayes
    :no-members:
    :no-inherited-members:

.. currentmodule:: skmultiflow

.. autosummary::
   :toctree: _autosummary

   bayes.naive_bayes

Ensemble methods
----------------

.. automodule:: skmultiflow.meta
   :no-members:
   :no-inherited-members:

.. currentmodule:: skmultiflow

.. autosummary::
   :toctree: _autosummary

   meta.adaptive_random_forests
   meta.leverage_bagging
   meta.oza_bagging
   meta.oza_bagging_adwin
   meta.multi_output_learner
   meta.classifier_chains
   meta.regressor_chains
   meta.batch_incremental
   meta.accuracy_weighted_ensemble
   meta.learn_pp
   meta.learn_nse

Lazy learning methods
---------------------

.. automodule:: skmultiflow.lazy
   :no-members:
   :no-inherited-members:

.. currentmodule:: skmultiflow

.. autosummary::
   :toctree: _autosummary

   lazy.kdtree
   lazy.knn
   lazy.knn_adwin
   lazy.sam_knn

Neural Networks
---------------

.. automodule:: skmultiflow.neural_networks
    :no-members:
    :no-inherited-members:

.. currentmodule:: skmultiflow

.. autosummary::
   :toctree: _autosummary

   neural_networks.perceptron

Tree methods
------------

.. automodule:: skmultiflow.trees
   :no-members:
   :no-inherited-members:

.. currentmodule:: skmultiflow

.. autosummary::
   :toctree: _autosummary

   trees.hoeffding_tree
   trees.hoeffding_adaptive_tree
   trees.hoeffding_anytime_tree
   trees.arf_hoeffding_tree
   trees.lc_hoeffding_tree
   trees.regression_hoeffding_tree
   trees.regression_hoeffding_adaptive_tree
   trees.multi_target_regression_hoeffding_tree

Drift Detection: :mod:`skmultiflow.drift_detection`
===================================================

.. automodule:: skmultiflow.drift_detection
   :no-members:
   :no-inherited-members:

.. currentmodule:: skmultiflow

.. autosummary::
   :toctree: _autosummary

   drift_detection.adwin
   drift_detection.ddm
   drift_detection.eddm
   drift_detection.page_hinkley

Evaluation: :mod:`skmultiflow.evaluation`
=========================================

.. automodule:: skmultiflow.evaluation
   :no-members:
   :no-inherited-members:

.. currentmodule:: skmultiflow

.. autosummary::
   :toctree: _autosummary

   evaluation.evaluate_holdout
   evaluation.evaluate_prequential