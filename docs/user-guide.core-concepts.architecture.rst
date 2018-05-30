============
Architecture
============

The :class:`~skmultiflow.core.base.StreamModel` class is the base class in ``scikit-multiflow``. It contains the following abstract methods:

* ``fit`` -- Trains a model in a batch fashion. Works as a an interface to batch methods that implement a ``fit()`` functions such as ``scikit-learn`` methods.
* ``partial_fit`` -- Incrementally trains a stream model.
* ``predict`` -- Predicts the target's value in supervised learning methods.
* ``predict_proba`` -- Calculates the probability of a sample pertaining to a given class in classification problems.

An ``StreamModel`` object interacts with two other objects: an :class:`~skmultiflow.data.base_stream.Stream` object and (optionally) an :class:`~skmultiflow.evaluation.base_evaluator.StreamEvaluator` object. The ``Stream`` object provides a continuous flow of data on request. The ``StreamEvaluator`` performs multiple tasks: query the stream for data, train and test the model on the incoming data and continuously tracks the model's performance.

Following, is the sequence to train a Stream Model and track performance in ``scikit-multiflow`` using the ``Prequential`` evaluator.

.. image:: _static/images/prequential_sequence.png
   :alt: prequential evaluation sequence
   :align: center