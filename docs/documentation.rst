=============
Documentation
=============

This is the API documentation for ``scikit-multiflow``.

Data: :mod:`skmultiflow.data`
=============================

.. automodule:: skmultiflow.data
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _autosummary

   skmultiflow.data.base_stream
   skmultiflow.data.data_stream
   skmultiflow.data.file_stream

Stream Generators
-----------------

.. autosummary::
   :toctree: _autosummary

   skmultiflow.data.agrawal_generator
   skmultiflow.data.hyper_plane_generator
   skmultiflow.data.led_generator
   skmultiflow.data.led_generator_drift
   skmultiflow.data.mixed_generator
   skmultiflow.data.random_rbf_generator
   skmultiflow.data.random_rbf_generator_drift
   skmultiflow.data.random_tree_generator
   skmultiflow.data.sea_generator
   skmultiflow.data.sine_generator
   skmultiflow.data.stagger_generator
   skmultiflow.data.waveform_generator
   skmultiflow.data.multilabel_generator
   skmultiflow.data.regression_generator

.. currentmodule:: skmultiflow

Learning methods
=================================================

Bayes methods
----------------

.. automodule:: skmultiflow.bayes
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _autosummary


   skmultiflow.bayes.naive_bayes

.. currentmodule:: skmultiflow

Ensemble methods
----------------

.. automodule:: skmultiflow.meta
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: _autosummary

   skmultiflow.meta.batch_incremental
   skmultiflow.meta.oza_bagging
   skmultiflow.meta.oza_bagging_adwin
   skmultiflow.meta.leverage_bagging
   skmultiflow.meta.adaptive_random_forests
   skmultiflow.meta.multi_output_learner
   skmultiflow.meta.classifier_chains
   skmultiflow.meta.regressor_chains

.. currentmodule:: skmultiflow

Lazy learning methods
---------------------

.. automodule:: skmultiflow.lazy
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: _autosummary

   skmultiflow.lazy.kdtree
   skmultiflow.lazy.knn
   skmultiflow.lazy.knn_adwin
   skmultiflow.lazy.sam_knn

.. currentmodule:: skmultiflow

Neural Networks
---------------

.. automodule:: skmultiflow.neural_networks
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _autosummary

   skmultiflow.neural_networks.perceptron

.. currentmodule:: skmultiflow

Tree methods
------------

.. automodule:: skmultiflow.trees
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: _autosummary

   skmultiflow.trees.hoeffding_tree
   skmultiflow.trees.hoeffding_adaptive_tree
   skmultiflow.trees.arf_hoeffding_tree
   skmultiflow.trees.lc_hoeffding_tree
   skmultiflow.trees.regression_hoeffding_tree
   skmultiflow.trees.regression_hoeffding_adaptive_tree

.. currentmodule:: skmultiflow

Drift Detection: :mod:`skmultiflow.drift_detection`
======================================================================

.. automodule:: skmultiflow.drift_detection
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: _autosummary

   skmultiflow.drift_detection.adwin
   skmultiflow.drift_detection.ddm
   skmultiflow.drift_detection.eddm
   skmultiflow.drift_detection.page_hinkley

.. currentmodule:: skmultiflow

Evaluation: :mod:`skmultiflow.evaluation`
=========================================

.. automodule:: skmultiflow.evaluation
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: _autosummary

   skmultiflow.evaluation.evaluate_holdout
   skmultiflow.evaluation.evaluate_prequential


Package, Indices and Search
===========================

.. toctree::
   :maxdepth: 1

   modules

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`