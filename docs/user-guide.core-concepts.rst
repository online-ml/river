=============
Core Concepts
=============

Consider a continuous stream of data :math:`A=\{(\vec{x}_t,y_t)\} | t = 1,\ldots,T` where :math:`T \rightarrow \infty`. :math:`\vec{x}_t` is a feature vector and :math:`y_t` the corresponding target where :math:`y` is continuous in the case of regression and discrete for classification. The objective is to predict the target :math:`y` for an unknown :math:`\vec{x}`. Two classes are considered in **binary** classification, :math:`y\in \{0,1\}`, while :math:`K>2` labels are used in **multi-label** classification, :math:`y\in \{1,\ldots,K\}`. For both *binary* and *multi-label* classification only one class is assigned per instance. On the other hand, in **multi-output** learning :math:`y` is a classes vector and :math:`\vec{x}_i` can be assigned multiple-classes at the same time.

Different to batch learning, where all data is available for training :math:`train(X, y)`; in stream learning, training is performed incrementally as new data is available :math:`train(\vec{x}_i, y_i)`. Performance :math:`P` of a given model is measured according to some loss function that evaluates the difference between the set of expected labels :math:`Y` and the predicted ones :math:`\hat{Y}`.

**Hold-out** evaluation is a popular performance evaluation method where tests are performed in a separate test set. **Prequential-evaluation** or *interleaved-test-then-train evaluation*, is a popular performance evaluation method for the stream setting, where tests are performed on new data before using it to train the model.

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

