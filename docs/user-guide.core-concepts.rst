=============
Core Concepts
=============

Consider a continuous stream of data :math:`A=\{(\vec{x}_t,y_t)\} | t = 1,\ldots,T` where :math:`T \rightarrow \infty`. :math:`\vec{x}_t` is a feature vector and :math:`y_t` the corresponding target where :math:`y` is continuous in the case of regression and discrete for classification. The objective is to predict the target :math:`y` for an unknown :math:`\vec{x}`. Two target_values are considered in **binary** classification, :math:`y\in \{0,1\}`, while :math:`K>2` labels are used in **multi-label** classification, :math:`y\in \{1,\ldots,K\}`. For both *binary* and *multi-label* classification only one class is assigned per instance. On the other hand, in **multi-output** learning :math:`y` is a target_values vector and :math:`\vec{x}_i` can be assigned multiple-target_values at the same time.

Different to batch learning, where all data is available for training :math:`train(X, y)`; in stream learning, training is performed incrementally as new data is available :math:`train(\vec{x}_i, y_i)`. Performance :math:`P` of a given model is measured according to some loss function that evaluates the difference between the set of expected labels :math:`Y` and the predicted ones :math:`\hat{Y}`.

**Hold-out** evaluation is a popular performance evaluation method where tests are performed in a separate test set. **Prequential-evaluation** or *interleaved-test-then-train evaluation*, is a popular performance evaluation method for the stream setting, where tests are performed on new data before using it to train the model.

.. toctree::
   :maxdepth: 1
   :caption: Read more:

   Architecture <user-guide.core-concepts.architecture>
   Stream class <user-guide.core-concepts.stream-class>
