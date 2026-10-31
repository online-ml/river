# Pipeline

A pipeline of estimators.

Pipelines allow you to chain different steps into a sequence. Typically, when doing supervised learning, a pipeline contains one or more transformation steps, whilst it's a regressor or a classifier. It is highly recommended to use pipelines with River. Indeed, in an online learning setting, it is very practical to have a model defined as a single object. Take a look at the [user guide](</recipes/pipelines>) for further information and practical examples.

One special thing to take notice to is the way transformers are handled. It is usual to predict something for a sample and wait for the ground truth to arrive. In such a scenario, the features are seen before the ground truth arrives. Therefore, the unsupervised parts of the pipeline are updated when `predict_one` and `predict_proba_one` are called. Usually the unsupervised parts of the pipeline are all the steps that precede the final step, which is a supervised model. However, some transformers are supervised and are therefore also updated during calls to `learn_one`.

## Examples

The recommended way to declare a pipeline is to use the `|` operator. The latter allows you to chain estimators in a very terse manner:

```python
from river import linear_model
from river import preprocessing

scaler = preprocessing.StandardScaler()
log_reg = linear_model.LinearRegression()
model = scaler | log_reg
```

This results in a pipeline that stores each step inside a dictionary.

```python
model
```

```text
Pipeline (
  StandardScaler (
    with_std=True
    window_size=None
  ),
  LinearRegression (
    optimizer=SGD (
      lr=Constant (
        learning_rate=0.01
      )
    )
    loss=Squared ()
    l2=0.
    l1=0.
    intercept_init=0.
    intercept_lr=Constant (
      learning_rate=0.01
    )
    clip_gradient=1e+12
    initializer=Zeros ()
  )
)
```

You can access parts of a pipeline in the same manner as a dictionary:

```python
model['LinearRegression']
```

```text
LinearRegression (
  optimizer=SGD (
    lr=Constant (
      learning_rate=0.01
    )
  )
  loss=Squared ()
  l2=0.
  l1=0.
  intercept_init=0.
  intercept_lr=Constant (
    learning_rate=0.01
  )
  clip_gradient=1e+12
  initializer=Zeros ()
)
```

Note that you can also declare a pipeline by using the [`compose.Pipeline`](<https://riverml.xyz/dev/api/compose/Pipeline>) constructor method, which is slightly more verbose:

```python
from river import compose

model = compose.Pipeline(scaler, log_reg)
```

By using a [`compose.TransformerUnion`](<https://riverml.xyz/dev/api/compose/TransformerUnion>), you can define complex pipelines that apply different steps to different parts of the data. For instance, we can extract word counts from text data, and extract polynomial features from numeric data.

```python
from river import feature_extraction as fx

tfidf = fx.TFIDF('text')
counts = fx.BagOfWords('text')
text_part = compose.Select('text') | (tfidf + counts)

num_part = compose.Select('a', 'b') | fx.PolynomialExtender()

model = text_part + num_part
model |= preprocessing.StandardScaler()
model |= linear_model.LinearRegression()
```

The following shows an example of using `debug_one` to visualize how the information flows and changes throughout the pipeline.

```python
from river import compose
from river import naive_bayes

dataset = [
    ('A positive comment', True),
    ('A negative comment', False),
    ('A happy comment', True),
    ('A lovely comment', True),
    ('A harsh comment', False)
]

tfidf = fx.TFIDF() | compose.Prefixer('tfidf_')
counts = fx.BagOfWords() | compose.Prefixer('count_')
mnb = naive_bayes.MultinomialNB()
model = (tfidf + counts) | mnb

for x, y in dataset:
    model.learn_one(x, y)

x = dataset[0][0]
report = model.debug_one(dataset[0][0])
print(report)
```

```text
0. Input
--------
A positive comment
1. Transformer union
--------------------
    1.0 TFIDF | Prefixer
    --------------------
    tfidf_comment: 0.43017 (float)
    tfidf_positive: 0.90275 (float)
    1.1 BagOfWords | Prefixer
    -------------------------
    count_comment: 1 (int)
    count_positive: 1 (int)
count_comment: 1 (int)
count_positive: 1 (int)
tfidf_comment: 0.43017 (float)
tfidf_positive: 0.90275 (float)
2. MultinomialNB
----------------
False: 0.19221
True: 0.80779
```

## Methods

> [!NOTE]
>
> **debug\_one**
>
> Displays the state of a set of features as it goes through the pipeline.
>
> **Parameters**
>
> - **x** — `dict`
> - **show\_types** — defaults to `True`
> - **n\_decimals** — defaults to `5`

> [!NOTE]
>
> **forecast**
>
> Return a forecast.
>
> Only works if each estimator has a `transform_one` method and the final estimator has a `forecast` method. This is the case of time series models from the `time_series` module.
>
> **Parameters**
>
> - **horizon** — `int`
> - **xs** — `list[dict] | None` — defaults to `None`

> [!NOTE]
>
> **learn\_many**
>
> Fit to a mini-batch.
>
> **Parameters**
>
> - **X** — `IntoDataFrame`
> - **y** — `IntoSeries | None` — defaults to `None`
> - **params**

> [!NOTE]
>
> **learn\_one**
>
> Fit to a single instance.
>
> **Parameters**
>
> - **x** — `dict`
> - **y** — defaults to `None`
> - **params**

> [!NOTE]
>
> **predict\_many**
>
> Call transform\_many, and then predict\_many on the final step.
>
> **Parameters**
>
> - **X** — `IntoDataFrame`

> [!NOTE]
>
> **predict\_one**
>
> Call `transform_one` on the first steps and `predict_one` on the last step.
>
> **Parameters**
>
> - **x** — `dict`
> - **params**

> [!NOTE]
>
> **predict\_proba\_many**
>
> Call transform\_many, and then predict\_proba\_many on the final step.
>
> **Parameters**
>
> - **X** — `IntoDataFrame`

> [!NOTE]
>
> **predict\_proba\_one**
>
> Call `transform_one` on the first steps and `predict_proba_one` on the last step.
>
> **Parameters**
>
> - **x** — `dict`
> - **params**

> [!NOTE]
>
> **score\_one**
>
> Call `transform_one` on the first steps and `score_one` on the last step.
>
> **Parameters**
>
> - **x** — `dict`
> - **params**

> [!NOTE]
>
> **transform\_many**
>
> Apply each transformer in the pipeline to some features.
>
> The final step in the pipeline will be applied if it is a transformer. If not, then it will be ignored and the output from the penultimate step will be returned. Note that the steps that precede the final step are assumed to all be transformers.
>
> **Parameters**
>
> - **X** — `IntoDataFrame`

> [!NOTE]
>
> **transform\_one**
>
> Apply each transformer in the pipeline to some features.
>
> The final step in the pipeline will be applied if it is a transformer. If not, then it will be ignored and the output from the penultimate step will be returned. Note that the steps that precede the final step are assumed to all be transformers.
>
> **Parameters**
>
> - **x** — `dict`
> - **params**
