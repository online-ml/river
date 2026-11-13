# The art of using pipelines

> [!NOTE]
>
> For a quick reference on pipeline syntax and features, see the [Pipelines recipe](<https://riverml.xyz/0.27.0/recipes/pipelines/index.md>).

Pipelines are a natural way to think about a machine learning system. Indeed with some practice a data scientist can visualise data "flowing" through a series of steps. The input is typically some raw data which has to be processed in some manner. The goal is to represent the data in such a way that is can be ingested by a machine learning algorithm. Along the way some steps will extract features, while others will normalize the data and remove undesirable elements. Pipelines are simple, and yet they are a powerful way of designing sophisticated machine learning systems.

Both [scikit-learn](<https://stackoverflow.com/questions/33091376/python-what-is-exactly-sklearn-pipeline-pipeline>) and [pandas](<https://tomaugspurger.github.io/method-chaining>) make it possible to use pipelines. However it's quite rare to see pipelines being used in practice (at least on Kaggle). Sometimes you get to see people using scikit-learn's `pipeline` module, however the `pipe` method from `pandas` is sadly underappreciated. A big reason why pipelines are not given much love is that it's easier to think of batch learning in terms of a script or a notebook. Indeed many people doing data science seem to prefer a procedural style to a declarative style. Moreover in practice pipelines can be a bit rigid if one wishes to do non-orthodox operations.

Although pipelines may be a bit of an odd fit for batch learning, they make complete sense when they are used for online learning. Indeed the UNIX philosophy has advocated the use of pipelines for data processing for many decades. If you can visualise data as a stream of observations then using pipelines should make a lot of sense to you. We'll attempt to convince you by writing a machine learning algorithm in a procedural way and then converting it to a declarative pipeline in small steps. Hopefully by the end you'll be convinced, or not!

In this notebook we'll manipulate data from the [Kaggle Recruit Restaurants Visitor Forecasting competition](<https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting>). The data is directly available through River's `datasets` module.

```python
from pprint import pprint
from river import datasets

for x, y in datasets.Restaurants():
    pprint(x)
    pprint(y)
    break
```

```text
Downloading https://maxhalford.github.io/files/datasets/kaggle_recruit_restaurants.zip (4.28 MiB)
Uncompressing into /home/runner/river_data/Restaurants
```

```text
{'area_name': 'Tōkyō-to Nerima-ku Toyotamakita',
 'date': datetime.datetime(2016, 1, 1, 0, 0),
 'genre_name': 'Izakaya',
 'is_holiday': True,
 'latitude': 35.7356234,
 'longitude': 139.6516577,
 'store_id': 'air_04341b588bde96cd'}
10
```

We'll start by building and running a model using a procedural coding style. The performance of the model doesn't matter, we're simply interested in the design of the model.

```python
from river import feature_extraction
from river import linear_model
from river import metrics
from river import preprocessing
from river import stats
from river import utils

means = (
    feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), 7)),
    feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), 14)),
    feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), 21))
)

scaler = preprocessing.StandardScaler()
lin_reg = linear_model.LinearRegression()
metric = metrics.MAE()

for x, y in datasets.Restaurants():

    # Derive date features
    x['weekday'] = x['date'].weekday()
    x['is_weekend'] = x['date'].weekday() in (5, 6)

    # Process the rolling means of the target  
    for mean in means:
        x = {**x, **mean.transform_one(x)}
        mean.learn_one(x, y)

    # Remove the key/value pairs that aren't features
    for key in ['store_id', 'date', 'genre_name', 'area_name', 'latitude', 'longitude']:
        x.pop(key)

    # Rescale the data
    scaler.learn_one(x)
    x = scaler.transform_one(x)

    # Fit the linear regression
    y_pred = lin_reg.predict_one(x)
    lin_reg.learn_one(x, y)

    # Update the metric using the out-of-fold prediction
    metric.update(y, y_pred)

print(metric)
```

```text
/tmp/ipykernel_4375/951201656.py:9: DeprecationWarning: Passing an instance to utils.Rolling is deprecated and will be removed in a future release. Pass the class and forward constructor kwargs instead, e.g. utils.Rolling(stats.Mean, window_size=...) or utils.Rolling(stats.Var, window_size=..., ddof=0).
  feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), 7)),
/tmp/ipykernel_4375/951201656.py:10: DeprecationWarning: Passing an instance to utils.Rolling is deprecated and will be removed in a future release. Pass the class and forward constructor kwargs instead, e.g. utils.Rolling(stats.Mean, window_size=...) or utils.Rolling(stats.Var, window_size=..., ddof=0).
  feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), 14)),
/tmp/ipykernel_4375/951201656.py:11: DeprecationWarning: Passing an instance to utils.Rolling is deprecated and will be removed in a future release. Pass the class and forward constructor kwargs instead, e.g. utils.Rolling(stats.Mean, window_size=...) or utils.Rolling(stats.Var, window_size=..., ddof=0).
  feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), 21))
```

```text
MAE: 8.316538
```

We're not using many features. We can print the last `x` to get an idea of the features (don't forget they've been scaled!)

```python
pprint(x)
```

```text
{'is_holiday': -0.23103573677646685,
 'is_weekend': 1.6249280076334165,
 'weekday': 1.0292832579142892,
 'y_mean_by_store_id': -1.3980979075298516}
```

The above chunk of code is quite explicit but it's a bit verbose. The whole point of libraries such as River is to make life easier for users. Moreover there's too much space for users to mess up the order in which things are done, which increases the chance of there being target leakage. We'll now rewrite our model in a declarative fashion using a pipeline *à la sklearn*.

```python
from river import compose


def get_date_features(x):
    weekday =  x['date'].weekday()
    return {'weekday': weekday, 'is_weekend': weekday in (5, 6)}


model = compose.Pipeline(
    ('features', compose.TransformerUnion(
        ('date_features', compose.FuncTransformer(get_date_features)),
        ('last_7_mean', feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), 7))),
        ('last_14_mean', feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), 14))),
        ('last_21_mean', feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), 21)))
    )),
    ('drop_non_features', compose.Discard('store_id', 'date', 'genre_name', 'area_name', 'latitude', 'longitude')),
    ('scale', preprocessing.StandardScaler()),
    ('lin_reg', linear_model.LinearRegression())
)

metric = metrics.MAE()

for x, y in datasets.Restaurants():

    # Make a prediction without using the target
    y_pred = model.predict_one(x)

    # Update the model using the target
    model.learn_one(x, y)

    # Update the metric using the out-of-fold prediction
    metric.update(y, y_pred)

print(metric)
```

```text
/tmp/ipykernel_4375/62508338.py:12: DeprecationWarning: Passing an instance to utils.Rolling is deprecated and will be removed in a future release. Pass the class and forward constructor kwargs instead, e.g. utils.Rolling(stats.Mean, window_size=...) or utils.Rolling(stats.Var, window_size=..., ddof=0).
  ('last_7_mean', feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), 7))),
/tmp/ipykernel_4375/62508338.py:13: DeprecationWarning: Passing an instance to utils.Rolling is deprecated and will be removed in a future release. Pass the class and forward constructor kwargs instead, e.g. utils.Rolling(stats.Mean, window_size=...) or utils.Rolling(stats.Var, window_size=..., ddof=0).
  ('last_14_mean', feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), 14))),
/tmp/ipykernel_4375/62508338.py:14: DeprecationWarning: Passing an instance to utils.Rolling is deprecated and will be removed in a future release. Pass the class and forward constructor kwargs instead, e.g. utils.Rolling(stats.Mean, window_size=...) or utils.Rolling(stats.Var, window_size=..., ddof=0).
  ('last_21_mean', feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), 21)))
```

```text
MAE: 8.413859
```

We use a `Pipeline` to arrange each step in a sequential order. A `TransformerUnion` is used to merge multiple feature extractors into a single transformer. The `for` loop is now much shorter and is thus easier to grok: we get the out-of-fold prediction, we fit the model, and finally we update the metric. This way of evaluating a model is typical of online learning, and so we put it wrapped it inside a function called `progressive_val_score` part of the `evaluate` module. We can use it to replace the `for` loop.

```python
from river import evaluate

model = compose.Pipeline(
    ('features', compose.TransformerUnion(
        ('date_features', compose.FuncTransformer(get_date_features)),
        ('last_7_mean', feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), 7))),
        ('last_14_mean', feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), 14))),
        ('last_21_mean', feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), 21)))
    )),
    ('drop_non_features', compose.Discard('store_id', 'date', 'genre_name', 'area_name', 'latitude', 'longitude')),
    ('scale', preprocessing.StandardScaler()),
    ('lin_reg', linear_model.LinearRegression())
)

evaluate.progressive_val_score(dataset=datasets.Restaurants(), model=model, metric=metrics.MAE())
```

```text
/tmp/ipykernel_4375/3293120273.py:6: DeprecationWarning: Passing an instance to utils.Rolling is deprecated and will be removed in a future release. Pass the class and forward constructor kwargs instead, e.g. utils.Rolling(stats.Mean, window_size=...) or utils.Rolling(stats.Var, window_size=..., ddof=0).
  ('last_7_mean', feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), 7))),
/tmp/ipykernel_4375/3293120273.py:7: DeprecationWarning: Passing an instance to utils.Rolling is deprecated and will be removed in a future release. Pass the class and forward constructor kwargs instead, e.g. utils.Rolling(stats.Mean, window_size=...) or utils.Rolling(stats.Var, window_size=..., ddof=0).
  ('last_14_mean', feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), 14))),
/tmp/ipykernel_4375/3293120273.py:8: DeprecationWarning: Passing an instance to utils.Rolling is deprecated and will be removed in a future release. Pass the class and forward constructor kwargs instead, e.g. utils.Rolling(stats.Mean, window_size=...) or utils.Rolling(stats.Var, window_size=..., ddof=0).
  ('last_21_mean', feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), 21)))
```

```
```

```text
MAE: [1;36m8.413859[0m
```

Notice that you couldn't have used the `progressive_val_score` method if you wrote the model in a procedural manner.

Our code is getting shorter, but it's still a bit difficult on the eyes. Indeed there is a lot of boilerplate code associated with pipelines that can get tedious to write. However River has some special tricks up it's sleeve to save you from a lot of pain.

The first trick is that the name of each step in the pipeline can be omitted. If no name is given for a step then River automatically infers one.

```python
model = compose.Pipeline(
    compose.TransformerUnion(
        compose.FuncTransformer(get_date_features),
        feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), 7)),
        feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), 14)),
        feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), 21))
    ),
    compose.Discard('store_id', 'date', 'genre_name', 'area_name', 'latitude', 'longitude'),
    preprocessing.StandardScaler(),
    linear_model.LinearRegression()
)

evaluate.progressive_val_score(datasets.Restaurants(), model, metrics.MAE())
```

```text
/tmp/ipykernel_4375/1516519861.py:4: DeprecationWarning: Passing an instance to utils.Rolling is deprecated and will be removed in a future release. Pass the class and forward constructor kwargs instead, e.g. utils.Rolling(stats.Mean, window_size=...) or utils.Rolling(stats.Var, window_size=..., ddof=0).
  feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), 7)),
/tmp/ipykernel_4375/1516519861.py:5: DeprecationWarning: Passing an instance to utils.Rolling is deprecated and will be removed in a future release. Pass the class and forward constructor kwargs instead, e.g. utils.Rolling(stats.Mean, window_size=...) or utils.Rolling(stats.Var, window_size=..., ddof=0).
  feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), 14)),
/tmp/ipykernel_4375/1516519861.py:6: DeprecationWarning: Passing an instance to utils.Rolling is deprecated and will be removed in a future release. Pass the class and forward constructor kwargs instead, e.g. utils.Rolling(stats.Mean, window_size=...) or utils.Rolling(stats.Var, window_size=..., ddof=0).
  feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), 21))
```

```
```

```text
MAE: [1;36m8.413859[0m
```

Under the hood a `Pipeline` inherits from `collections.OrderedDict`. Indeed this makes sense because if you think about it a `Pipeline` is simply a sequence of steps where each step has a name. The reason we mention this is because it means you can manipulate a `Pipeline` the same way you would manipulate an ordinary `dict`. For instance we can print the name of each step by using the `keys` method.

```python
for name in model.steps:
    print(name)
```

```text
TransformerUnion
Discard
StandardScaler
LinearRegression
```

The first step is a `FeatureUnion` and it's string representation contains the string representation of each of it's elements. Not having to write names saves up some time and space and is certainly less tedious.

The next trick is that we can use mathematical operators to compose our pipeline. For example we can use the `+` operator to merge `Transformer`s into a `TransformerUnion`.

```python
model = compose.Pipeline(
    compose.FuncTransformer(get_date_features) + \
    feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), 7)) + \
    feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), 14)) + \
    feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), 21)),

    compose.Discard('store_id', 'date', 'genre_name', 'area_name', 'latitude', 'longitude'),
    preprocessing.StandardScaler(),
    linear_model.LinearRegression()
)

evaluate.progressive_val_score(datasets.Restaurants(), model, metrics.MAE())
```

```text
/tmp/ipykernel_4375/4037441292.py:3: DeprecationWarning: Passing an instance to utils.Rolling is deprecated and will be removed in a future release. Pass the class and forward constructor kwargs instead, e.g. utils.Rolling(stats.Mean, window_size=...) or utils.Rolling(stats.Var, window_size=..., ddof=0).
  feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), 7)) + \
/tmp/ipykernel_4375/4037441292.py:4: DeprecationWarning: Passing an instance to utils.Rolling is deprecated and will be removed in a future release. Pass the class and forward constructor kwargs instead, e.g. utils.Rolling(stats.Mean, window_size=...) or utils.Rolling(stats.Var, window_size=..., ddof=0).
  feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), 14)) + \
/tmp/ipykernel_4375/4037441292.py:5: DeprecationWarning: Passing an instance to utils.Rolling is deprecated and will be removed in a future release. Pass the class and forward constructor kwargs instead, e.g. utils.Rolling(stats.Mean, window_size=...) or utils.Rolling(stats.Var, window_size=..., ddof=0).
  feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), 21)),
```

```
```

```text
MAE: [1;36m8.413859[0m
```

Likewise we can use the `|` operator to assemble steps into a `Pipeline`.

```python
model = (
    compose.FuncTransformer(get_date_features) +
    feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), 7)) +
    feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), 14)) +
    feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), 21))
)

to_discard = ['store_id', 'date', 'genre_name', 'area_name', 'latitude', 'longitude']

model = model | compose.Discard(*to_discard) | preprocessing.StandardScaler()

model |= linear_model.LinearRegression()

evaluate.progressive_val_score(datasets.Restaurants(), model, metrics.MAE())
```

```text
/tmp/ipykernel_4375/2780677158.py:3: DeprecationWarning: Passing an instance to utils.Rolling is deprecated and will be removed in a future release. Pass the class and forward constructor kwargs instead, e.g. utils.Rolling(stats.Mean, window_size=...) or utils.Rolling(stats.Var, window_size=..., ddof=0).
  feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), 7)) +
/tmp/ipykernel_4375/2780677158.py:4: DeprecationWarning: Passing an instance to utils.Rolling is deprecated and will be removed in a future release. Pass the class and forward constructor kwargs instead, e.g. utils.Rolling(stats.Mean, window_size=...) or utils.Rolling(stats.Var, window_size=..., ddof=0).
  feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), 14)) +
/tmp/ipykernel_4375/2780677158.py:5: DeprecationWarning: Passing an instance to utils.Rolling is deprecated and will be removed in a future release. Pass the class and forward constructor kwargs instead, e.g. utils.Rolling(stats.Mean, window_size=...) or utils.Rolling(stats.Var, window_size=..., ddof=0).
  feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), 21))
```

```
```

```text
MAE: [1;36m8.413859[0m
```

Hopefully you'll agree that this is a powerful way to express machine learning pipelines. For some people this should be quite remeniscent of the UNIX pipe operator. One final trick we want to mention is that functions are automatically wrapped with a `FuncTransformer`, which can be quite handy.

```python
model = get_date_features

for n in [7, 14, 21]:
    model += feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), n))

model |= compose.Discard(*to_discard)
model |= preprocessing.StandardScaler()
model |= linear_model.LinearRegression()

evaluate.progressive_val_score(datasets.Restaurants(), model, metrics.MAE())
```

```text
/tmp/ipykernel_4375/3016371111.py:4: DeprecationWarning: Passing an instance to utils.Rolling is deprecated and will be removed in a future release. Pass the class and forward constructor kwargs instead, e.g. utils.Rolling(stats.Mean, window_size=...) or utils.Rolling(stats.Var, window_size=..., ddof=0).
  model += feature_extraction.TargetAgg(by='store_id', how=utils.Rolling(stats.Mean(), n))
```

```
```

```text
MAE: [1;36m8.413859[0m
```

Naturally some may prefer the procedural style we first used because they find it easier to work with. It all depends on your style and you should use what you feel comfortable with. However we encourage you to use operators because we believe that this will increase the readability of your code, which is very important. To each their own!

Before finishing we can take an interactive look at our pipeline.

```python
model
```

```
```

```
╭─────────────────────────────── Traceback (most recent call last) ────────────────────────────────╮
│ /home/runner/work/river/river/.venv/lib/python3.13/site-packages/IPython/core/formatters.py:282  │
│ in catch_format_error                                                                            │
│                                                                                                  │
│    279 def catch_format_error(method, self, *args, **kwargs):                                    │
│    280 │   """show traceback on failed format call"""                                            │
│    281 │   try:                                                                                  │
│ ❱  282 │   │   r = method(self, *args, **kwargs)                                                 │
│    283 │   except NotImplementedError:                                                           │
│    284 │   │   # don't warn on NotImplementedErrors                                              │
│    285 │   │   return self._check_return(None, args[0])                                          │
│                                                                                                  │
│ /home/runner/work/river/river/.venv/lib/python3.13/site-packages/IPython/core/formatters.py:406  │
│ in __call__                                                                                      │
│                                                                                                  │
│    403 │   │   │   # Finally look for special method names                                       │
│    404 │   │   │   method = get_real_method(obj, self.print_method)                              │
│    405 │   │   │   if method is not None:                                                        │
│ ❱  406 │   │   │   │   return method()                                                           │
│    407 │   │   │   return None                                                                   │
│    408 │   │   else:                                                                             │
│    409 │   │   │   return None                                                                   │
│                                                                                                  │
│ /home/runner/work/river/river/river/base/estimator.py:65 in _repr_html_                          │
│                                                                                                  │
│    62 │   │                                                                                      │
│    63 │   │   from river.base import viz                                                         │
│    64 │   │                                                                                      │
│ ❱  65 │   │   div = viz.to_html(self)                                                            │
│    66 │   │   div_str = ET.tostring(div, encoding="unicode")                                     │
│    67 │   │   return f"<div>{div_str}<style scoped>{viz.CSS}</style></div>"                      │
│    68                                                                                            │
│                                                                                                  │
│ /home/runner/work/river/river/river/base/viz.py:13 in to_html                                    │
│                                                                                                  │
│    10                                                                                            │
│    11 def to_html(obj: [base.Estimator](../api/base/Estimator)) -> ET.Element:                                            │
│    12 │   if isinstance(obj, [compose.Pipeline](../api/compose/Pipeline)):                                                  │
│ ❱  13 │   │   return pipeline_to_html(obj)                                                       │
│    14 │   if isinstance(obj, [compose.TransformerUnion](../api/compose/TransformerUnion)):                                          │
│    15 │   │   return union_to_html(obj)                                                          │
│    16 │   if isinstance(obj, [base.Wrapper](../api/base/Wrapper)):                                                      │
│                                                                                                  │
│ /home/runner/work/river/river/river/base/viz.py:51 in pipeline_to_html                           │
│                                                                                                  │
│    48 │   div = ET.Element("div", attrib={"class": "river-component river-pipeline"})            │
│    49 │                                                                                          │
│    50 │   for step in pipeline.steps.values():                                                   │
│ ❱  51 │   │   div.append(to_html(step))                                                          │
│    52 │                                                                                          │
│    53 │   return div                                                                             │
│    54                                                                                            │
│                                                                                                  │
│ /home/runner/work/river/river/river/base/viz.py:15 in to_html                                    │
│                                                                                                  │
│    12 │   if isinstance(obj, [compose.Pipeline](../api/compose/Pipeline)):                                                  │
│    13 │   │   return pipeline_to_html(obj)                                                       │
│    14 │   if isinstance(obj, [compose.TransformerUnion](../api/compose/TransformerUnion)):                                          │
│ ❱  15 │   │   return union_to_html(obj)                                                          │
│    16 │   if isinstance(obj, [base.Wrapper](../api/base/Wrapper)):                                                      │
│    17 │   │   return wrapper_to_html(obj)                                                        │
│    18 │   return estimator_to_html(obj)                                                          │
│                                                                                                  │
│ /home/runner/work/river/river/river/base/viz.py:70 in union_to_html                              │
│                                                                                                  │
│    67 │   summary.append(pre)                                                                    │
│    68 │                                                                                          │
│    69 │   for transformer in union.transformers.values():                                        │
│ ❱  70 │   │   div.append(to_html(transformer))                                                   │
│    71 │                                                                                          │
│    72 │   return div                                                                             │
│    73                                                                                            │
│                                                                                                  │
│ /home/runner/work/river/river/river/base/viz.py:18 in to_html                                    │
│                                                                                                  │
│    15 │   │   return union_to_html(obj)                                                          │
│    16 │   if isinstance(obj, [base.Wrapper](../api/base/Wrapper)):                                                      │
│    17 │   │   return wrapper_to_html(obj)                                                        │
│ ❱  18 │   return estimator_to_html(obj)                                                          │
│    19                                                                                            │
│    20                                                                                            │
│    21 def estimator_to_html(estimator: [base.Estimator](../api/base/Estimator)) -> ET.Element:                            │
│                                                                                                  │
│ /home/runner/work/river/river/river/base/viz.py:41 in estimator_to_html                          │
│                                                                                                  │
│    38 │   if isinstance(estimator, [compose.FuncTransformer](../api/compose/FuncTransformer)):                                     │
│    39 │   │   code.text = f"\n{inspect.getsource(estimator.func)}\n"                             │
│    40 │   else:                                                                                  │
│ ❱  41 │   │   code.text = f"{repr(estimator)}\n"                                                 │
│    42 │   details.append(code)                                                                   │
│    43 │                                                                                          │
│    44 │   return details                                                                         │
│                                                                                                  │
│ /home/runner/work/river/river/river/base/base.py:71 in __repr__                                  │
│                                                                                                  │
│    68 │   │   return self.__class__.__name__                                                     │
│    69 │                                                                                          │
│    70 │   def __repr__(self) -> str:                                                             │
│ ❱  71 │   │   return _repr_obj(obj=self)                                                         │
│    72 │                                                                                          │
│    73 │   @classmethod                                                                           │
│    74 │   def _unit_test_params(cls) -> collections.abc.Iterator[dict[str, [utils.random.Any](../api/utils/random/Any)]]:         │
│                                                                                                  │
│ /home/runner/work/river/river/river/base/base.py:576 in _repr_obj                                │
│                                                                                                  │
│   573 │   │   elif isinstance(val, set):                                                         │
│   574 │   │   │   val = sorted(val)                                                              │
│   575 │   │   elif hasattr(val, "__class__") and "river." in str(type(val)):                     │
│ ❱ 576 │   │   │   val = _repr_obj(obj=val, show_modules=show_modules, depth=depth + 1)           │
│   577 │   │                                                                                      │
│   578 │   │   rep += f"\n{tab * (depth + 1)}{name}={val}"                                        │
│   579                                                                                            │
│                                                                                                  │
│ /home/runner/work/river/river/river/base/base.py:547 in _repr_obj                                │
│                                                                                                  │
│   544 │   tab = "\t"                                                                             │
│   545 │                                                                                          │
│   546 │   params = {                                                                             │
│ ❱ 547 │   │   name: getattr(obj, name)                                                           │
│   548 │   │   for name, param in inspect.signature(obj.__init__).parameters.items()              │
│   549 │   │   if not (                                                                           │
│   550 │   │   │   param.name == "args"                                                           │
│                                                                                                  │
│ /home/runner/work/river/river/river/utils/rolling.py:41 in __getattr__                           │
│                                                                                                  │
│    38 │   │   │   obj = object.__getattribute__(self, "obj")                                     │
│    39 │   │   except AttributeError:                                                             │
│    40 │   │   │   raise AttributeError(name)                                                     │
│ ❱  41 │   │   return getattr(obj, name)                                                          │
│    42 │                                                                                          │
│    43 │   def __getitem__(self, idx: Any) -> object:                                             │
│    44 │   │   # Enable for when it needs, throws a runtime error as usual if tried on a type t   │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
AttributeError: 'Mean' object has no attribute 'cls'
```

```text
[1m<[0m[1;95mrepr-error[0m[39m [0m[32m"'Mean' object has no attribute 'cls'"[0m[1m>[0m
```
