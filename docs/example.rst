Quick example
=============

In the following example we'll use a linear regression to forecast the number of available bikes in [bike stations](https://www.wikiwand.com/en/Bicycle-sharing_system) from the city of Toulouse. Each observation looks like this:

.. code-block:: python


    >>> import pprint
    >>> from creme import datasets

    >>> X_y = datasets.ToulouseBikes()
    >>> x, y = next(iter(X_y))

    >>> pprint.pprint(x)
    {'clouds': 75,
    'description': 'light rain',
    'humidity': 81,
    'moment': datetime.datetime(2016, 4, 1, 0, 0, 7),
    'pressure': 1017.0,
    'station': 'metro-canal-du-midi',
    'temperature': 6.54,
    'wind': 9.3}

    >>> print(f'Number of bikes: {y}')
    Number of bikes: 1

We will include all the available numeric features in our model. We will also use target encoding by calculating a running average of the target per station and hour. Before being fed to the linear regression, the features will be scaled using a `StandardScaler`. Note that each of these steps works in a streaming fashion, including the feature extraction. We'll evaluate the model by asking it to forecast 30 minutes ahead while delaying the true answers, which ensures that we're simulating a production scenario. Finally we will print the current score every 20,000 predictions.

.. code-block:: python

    >>> import datetime as dt
    >>> from creme import compose
    >>> from creme import datasets
    >>> from creme import feature_extraction
    >>> from creme import linear_model
    >>> from creme import metrics
    >>> from creme import model_selection
    >>> from creme import preprocessing
    >>> from creme import stats

    >>> X_y = datasets.ToulouseBikes()

    >>> def add_hour(x):
    ...     x['hour'] = x['moment'].hour
    ...     return x

    >>> model = compose.Whitelister('clouds', 'humidity', 'pressure', 'temperature', 'wind')
    >>> model += (
    ...     add_hour |
    ...     feature_extraction.TargetAgg(by=['station', 'hour'], how=stats.Mean())
    ... )
    >>> model += feature_extraction.TargetAgg(by='station', how=stats.EWMean(0.5))
    >>> model |= preprocessing.StandardScaler()
    >>> model |= linear_model.LinearRegression()

    >>> model_selection.progressive_val_score(
    ...     X_y=X_y,
    ...     model=model,
    ...     metric=metrics.MAE(),
    ...     on='moment',
    ...     delay=dt.timedelta(minutes=30),
    ...     print_every=30_000
    ... )
    [30,000] MAE: 2.230049
    [60,000] MAE: 2.290409
    [90,000] MAE: 2.334638
    [120,000] MAE: 2.315149
    [150,000] MAE: 2.319982
    [180,000] MAE: 2.335385
    MAE: 2.338837

You can visualize the pipeline as so:

.. code-block:: python

    >>> model
    Pipeline (
    TransformerUnion (
        Whitelister (
        whitelist=('clouds', 'humidity', 'pressure', 'temperature', 'wind')
        ),
        Pipeline (
        FuncTransformer (
            func="add_hour"
        ),
        TargetAgg (
            by=['station', 'hour']
            how=Mean ()
            target_name="target"
        )
        ),
        TargetAgg (
        by=['station']
        how=EWMean (
            alpha=0.5
        )
        target_name="target"
        )
    ),
    StandardScaler (
        with_mean=True
        with_std=True
    ),
    LinearRegression (
        optimizer=SGD (
        lr=InverseScaling (
            learning_rate=0.01
            power=0.25
        )
        )
        loss=Squared ()
        l2=0.
        intercept=9.742884
        intercept_lr=Constant (
        learning_rate=0.01
        )
        clip_gradient=1e+12
        initializer=Zeros ()
    )
    )

You can also obtain a graphical representation of the pipeline.

.. code-block:: python

    >>> dot = model.draw()

<div align="center">
  <img src="_static/bikes_pipeline.svg" alt="bikes_pipeline"/>
</div>
