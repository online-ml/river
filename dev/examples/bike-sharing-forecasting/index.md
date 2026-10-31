# Bike-sharing forecasting

In this tutorial we're going to forecast the number of bikes in 5 bike stations from the city of Toulouse. We'll do so by building a simple model step by step. The dataset contains 182,470 observations. Let's first take a peek at the data.

```python
from pprint import pprint
from river import datasets

dataset = datasets.Bikes()

for x, y in dataset:
    pprint(x)
    print(f'Number of available bikes: {y}')
    break
```

```text
{'clouds': 75,
 'description': 'light rain',
 'humidity': 81,
 'moment': datetime.datetime(2016, 4, 1, 0, 0, 7),
 'pressure': 1017.0,
 'station': 'metro-canal-du-midi',
 'temperature': 6.54,
 'wind': 9.3}
Number of available bikes: 1
```

Let's start by using a simple linear regression on the numeric features. We can select the numeric features and discard the rest of the features using a `Select`. Linear regression is very likely to go haywire if we don't scale the data, so we'll use a `StandardScaler` to do just that. We'll evaluate the model by measuring the mean absolute error. Finally we'll print the score every 20,000 observations.

```python
from river import compose
from river import linear_model
from river import metrics
from river import evaluate
from river import preprocessing
from river import optim

model = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
model |= preprocessing.StandardScaler()
model |= linear_model.LinearRegression(optimizer=optim.SGD(0.001))

metric = metrics.MAE()

evaluate.progressive_val_score(dataset, model, metric, print_every=20_000)
```

```text
[20,000] MAE: 4.912763
[40,000] MAE: 5.333578
[60,000] MAE: 5.330969
[80,000] MAE: 5.392334
[100,000] MAE: 5.423078
[120,000] MAE: 5.541239
[140,000] MAE: 5.613038
[160,000] MAE: 5.622441
[180,000] MAE: 5.567836
[182,470] MAE: 5.563905
```

```text
MAE: 5.563905
```

The model doesn't seem to be doing that well, but then again we didn't provide a lot of features. Generally, a good idea for this kind of problem is to look at an average of the previous values. For example, for each station we can look at the average number of bikes per hour. To do so we first have to extract the hour from the `moment` field. We can then use a `TargetAgg` to aggregate the values of the target.

```python
from river import feature_extraction
from river import stats

def get_hour(x):
    x['hour'] = x['moment'].hour
    return x

model = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
model += (
    get_hour |
    feature_extraction.TargetAgg(by=['station', 'hour'], how=stats.Mean())
)
model |= preprocessing.StandardScaler()
model |= linear_model.LinearRegression(optimizer=optim.SGD(0.001))

metric = metrics.MAE()

evaluate.progressive_val_score(dataset, model, metric, print_every=20_000)
```

```text
[20,000] MAE: 3.720766
[40,000] MAE: 3.829739
[60,000] MAE: 3.844905
[80,000] MAE: 3.910137
[100,000] MAE: 3.888553
[120,000] MAE: 3.923644
[140,000] MAE: 3.980882
[160,000] MAE: 3.949972
[180,000] MAE: 3.934489
[182,470] MAE: 3.933442
```

```text
MAE: 3.933442
```

By adding a single feature, we've managed to significantly reduce the mean absolute error. At this point you might think that the model is getting slightly complex, and is difficult to understand and test. Pipelines have the advantage of being terse, but they aren't always to debug. Thankfully River has some ways to relieve the pain.

The first thing we can do it to visualize the pipeline, to get an idea of how the data flows through it.

```python
model
```

**```
['clouds', [...]
```**

`Select (   clouds   humidity   pressure   temperature   wind ) `

**```
get_hour
```**

`  def get_hour(x):     x['hour'] = x['moment'].hour     return x   `

**```
y_mean_by_station_and_hour
```**

`TargetAgg (   by=['station', 'hour']   how=Mean ()   target_name="y" ) `

**```
StandardScaler
```**

`StandardScaler (   with_std=True ) `

**```
LinearRegression
```**

`LinearRegression (   optimizer=SGD (     lr=Constant (       learning_rate=0.001     )   )   loss=Squared ()   l2=0.   l1=0.   intercept_init=0.   intercept_lr=Constant (     learning_rate=0.01   )   clip_gradient=1e+12   initializer=Zeros () ) `

We can also use the `debug_one` method to see what happens to one particular instance. Let's train the model on the first 10,000 observations and then call `debug_one` on the next one. To do this, we will turn the `Bike` object into a Python generator with `iter()` function. The Pythonic way to read the first 10,000 elements of a generator is to use `itertools.islice`.

```python
import itertools

model = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
model += (
    get_hour |
    feature_extraction.TargetAgg(by=['station', 'hour'], how=stats.Mean())
)
model |= preprocessing.StandardScaler()
model |= linear_model.LinearRegression()

for x, y in itertools.islice(dataset, 10000):
    y_pred = model.predict_one(x)
    model.learn_one(x, y)

x, y = next(iter(dataset))
print(model.debug_one(x))
```

```text
0. Input
--------
clouds: 75 (int)
description: light rain (str)
humidity: 81 (int)
moment: 2016-04-01 00:00:07 (datetime)
pressure: 1,017.00000 (float)
station: metro-canal-du-midi (str)
temperature: 6.54000 (float)
wind: 9.30000 (float)

1. Transformer union
--------------------
    1.0 Select
    ----------
    clouds: 75 (int)
    humidity: 81 (int)
    pressure: 1,017.00000 (float)
    temperature: 6.54000 (float)
    wind: 9.30000 (float)

    1.1 get_hour | y_mean_by_station_and_hour
    -----------------------------------------
    y_mean_by_station_and_hour: 4.43243 (float)

clouds: 75 (int)
humidity: 81 (int)
pressure: 1,017.00000 (float)
temperature: 6.54000 (float)
wind: 9.30000 (float)
y_mean_by_station_and_hour: 4.43243 (float)

2. StandardScaler
-----------------
clouds: 0.47566 (float)
humidity: 0.42247 (float)
pressure: 1.05314 (float)
temperature: -1.22098 (float)
wind: 2.21104 (float)
y_mean_by_station_and_hour: -0.59098 (float)

3. LinearRegression
-------------------
Name                         Value      Weight     Contribution  
                 Intercept    1.00000    6.58252        6.58252  
                  pressure    1.05314    3.78529        3.98646  
                  humidity    0.42247    1.44921        0.61225  
y_mean_by_station_and_hour   -0.59098    0.54167       -0.32011  
                    clouds    0.47566   -1.92255       -0.91448  
                      wind    2.21104   -0.77720       -1.71843  
               temperature   -1.22098    2.47030       -3.01619  

Prediction: 5.21201
```

The `debug_one` method shows what happens to an input set of features, step by step.

And now comes the catch. Up until now we've been using the `progressive_val_score` method from the `evaluate` module. What this does it that it sequentially predicts the output of an observation and updates the model immediately afterwards. This way of proceeding is often used for evaluating online learning models. But in some cases it is the wrong approach.

When evaluating a machine learning model, the goal is to simulate production conditions in order to get a trust-worthy assessment of the performance of the model. In our case, we typically want to forecast the number of bikes available in a station, say, 30 minutes ahead. Then, once the 30 minutes have passed, the true number of available bikes will be available and we will be able to update the model using the features available 30 minutes ago.

What we really want is to evaluate the model by forecasting 30 minutes ahead and only updating the model once the true values are available. This can be done using the `moment` and `delay` parameters in the `progressive_val_score` method. The idea is that each observation in the stream of the data is shown twice to the model: once for making a prediction, and once for updating the model when the true value is revealed. The `moment` parameter determines which variable should be used as a timestamp, while the `delay` parameter controls the duration to wait before revealing the true values to the model.

```python
import datetime as dt

evaluate.progressive_val_score(
    dataset=dataset,
    model=model.clone(),
    metric=metrics.MAE(),
    moment='moment',
    delay=dt.timedelta(minutes=30),
    print_every=20_000
)
```

```text
[20,000] MAE: 20.198137
[40,000] MAE: 12.199763
[60,000] MAE: 9.468279
[80,000] MAE: 8.126625
[100,000] MAE: 7.273133
[120,000] MAE: 6.735469
[140,000] MAE: 6.376704
[160,000] MAE: 6.06156
[180,000] MAE: 5.806744
[182,470] MAE: 5.780772
```

```text
MAE: 5.780772
```

The performance is a bit worse, which is to be expected. Indeed, the task is more difficult: the model is only shown the ground truth 30 minutes after making a prediction.

The takeaway of this notebook is that the `progressive_val_score` method can be used to simulate a production scenario, and is thus extremely valuable.
