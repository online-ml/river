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

.. currentmodule:: skmultiflow

Stream Generators
-----------------

.. automodule:: skmultiflow.data.generators
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: _autosummary

   skmultiflow.data.generators.agrawal_generator
   skmultiflow.data.generators.hyper_plane_generator
   skmultiflow.data.generators.led_generator
   skmultiflow.data.generators.led_generator_drift
   skmultiflow.data.generators.mixed_generator
   skmultiflow.data.generators.random_rbf_generator
   skmultiflow.data.generators.random_rbf_generator_drift
   skmultiflow.data.generators.random_tree_generator
   skmultiflow.data.generators.sea_generator
   skmultiflow.data.generators.sine_generator
   skmultiflow.data.generators.stagger_generator
   skmultiflow.data.generators.waveform_generator
   skmultiflow.data.generators.multilabel_generator
   skmultiflow.data.generators.regression_generator

.. currentmodule:: skmultiflow

Classification: :mod:`skmultiflow.classification`
=================================================

.. automodule:: skmultiflow.classification
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _autosummary

   skmultiflow.classification.multi_output_learner
   skmultiflow.classification.naive_bayes
   skmultiflow.classification.perceptron

.. currentmodule:: skmultiflow

Ensemble methods
----------------

.. automodule:: skmultiflow.classification.meta
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: _autosummary

   skmultiflow.classification.meta.batch_incremental
   skmultiflow.classification.meta.oza_bagging
   skmultiflow.classification.meta.oza_bagging_adwin
   skmultiflow.classification.meta.leverage_bagging
   skmultiflow.classification.meta.adaptive_random_forests

.. currentmodule:: skmultiflow

Tree methods
------------

.. automodule:: skmultiflow.classification.trees
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: _autosummary

   skmultiflow.classification.trees.hoeffding_tree
   skmultiflow.classification.trees.hoeffding_adaptive_tree
   skmultiflow.classification.trees.arf_hoeffding_tree

.. currentmodule:: skmultiflow

Lazy learning methods
-------------------------

.. automodule:: skmultiflow.classification.lazy
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: _autosummary

   skmultiflow.classification.lazy.knn
   skmultiflow.classification.lazy.knn_adwin
   skmultiflow.classification.lazy.sam_knn

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