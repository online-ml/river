Quick example
=============

In the following example, we will train a logistic regression to predict whether or not the price of electricity will go up or down in the next 30 minutes. We will be using a real world dataset of electricity prices from New South Wales in Australia. The dataset can be streamed by using the :func:`creme.datasets.fetch_electricity` method. Here is what the first observation looks like:

.. code-block:: python

    >>> from creme import datasets
    >>> X_y = datasets.fetch_electricity()
    >>> x, y = next(X_y)

    >>> x
    {'date': 0.0,
     'day': 2,
     'period': 0.0,
     'nswprice': 0.056443,
     'nswdemand': 0.439155,
     'vicprice': 0.003467,
     'vicdemand': 0.422915,
     'transfer': 0.414912}

    >>> y
    True

For each observation in the dataset, we will call ``predict_one`` to obtain the output predicted by the model. The `metrics.Accuracy` metric can then be updated by providing it with the true output ``y`` and the predicted output ``y_pred``. Finally, the model can be updated by calling ``fit_one``.

.. code-block:: python

    >>> from creme import datasets
    >>> from creme import linear_model
    >>> from creme import metrics
    >>> from creme import optim
    >>> from creme import preprocessing

    >>> X_y = datasets.fetch_electricity()

    >>> model = preprocessing.StandardScaler()
    >>> model |= linear_model.LogisticRegression(optimizer=optim.SGD(.1))

    >>> metric = metrics.Accuracy()

    >>> for x, y in X_y:
    ...     y_pred = model.predict_one(x)  # Make a prediction
    ...     metric = metric.update(y, y_pred)  # Update the metric
    ...     model = model.fit_one(x, y)  # Update the model

    >>> print(metric)
    Accuracy: 0.894664

``creme`` has a lot to offer, and as such we invite you to check out :doc:`API reference <api>` as well as the :doc:`user guide <user-guide>` for more information.
