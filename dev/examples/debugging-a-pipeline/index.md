# Debugging a pipeline

River encourages users to make use of pipelines. The biggest pain point of pipelines is that it can be hard to understand what's happening to the data, especially when the pipeline is complex. Fortunately the `Pipeline` class has a `debug_one` method that can help out.

Let's look at a fairly complex pipeline for predicting the number of bikes in 5 bike stations from the city of Toulouse. It doesn't matter if you understand the pipeline or not; the point of this notebook is to learn how to introspect a pipeline.

```python
import datetime as dt
from river import compose
from river import datasets
from river import feature_extraction
from river import linear_model
from river import metrics
from river import preprocessing
from river import stats
from river import stream


X_y = datasets.Bikes()
X_y = stream.simulate_qa(X_y, moment='moment', delay=dt.timedelta(minutes=30))

def add_time_features(x):
    return {
        **x,
        'hour': x['moment'].hour,
        'day': x['moment'].weekday()
    }

model = add_time_features
model |= (
    compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind') +
    feature_extraction.TargetAgg(by=['station', 'hour'], how=stats.Mean()) +
    feature_extraction.TargetAgg(by='station', how=stats.EWMean())
)
model |= preprocessing.StandardScaler()
model |= linear_model.LinearRegression()

metric = metrics.MAE()

questions = {}

for i, x, y in X_y:
    # Question
    is_question = y is None
    if is_question:
        y_pred = model.predict_one(x)
        questions[i] = y_pred

    # Answer
    else:
        metric.update(y, questions[i])
        model.learn_one(x, y)

        if i >= 30000 and i % 30000 == 0:
            print(i, metric)
```

```text
30000 MAE: 13.328051
60000 MAE: 7.824087
90000 MAE: 6.003909
120000 MAE: 5.052855
150000 MAE: 4.496826
180000 MAE: 4.140702
```

Let's start by looking at the pipeline. You can click each cell to display the current state for each step of the pipeline.

```python
model
```

**```
add_time_features
```**

`  def add_time_features(x):     return {         **x,         'hour': x['moment'].hour,         'day': x['moment'].weekday()     }   `

**```
['clouds', [...]
```**

`Select (   clouds   humidity   pressure   temperature   wind ) `

**```
y_mean_by_station_and_hour
```**

`TargetAgg (   by=['station', 'hour']   how=Mean ()   target_name="y" ) `

**```
y_ewm_0.5_by_station
```**

`TargetAgg (   by=['station']   how=EWMean (     fading_factor=0.5   )   target_name="y" ) `

**```
StandardScaler
```**

`StandardScaler (   with_std=True ) `

**```
LinearRegression
```**

`LinearRegression (   optimizer=SGD (     lr=Constant (       learning_rate=0.01     )   )   loss=Squared ()   l2=0.   l1=0.   intercept_init=0.   intercept_lr=Constant (     learning_rate=0.01   )   clip_gradient=1e+12   initializer=Zeros () ) `

As mentioned above the `Pipeline` class has a `debug_one` method. You can use this at any point you want to visualize what happen to an input `x`. For example, let's see what happens to the last seen `x`.

```python
print(model.debug_one(x))
```

```text
0. Input
--------
clouds: 88 (int)
description: overcast clouds (str)
humidity: 84 (int)
moment: 2016-10-05 09:57:18 (datetime)
pressure: 1,017.34000 (float)
station: pomme (str)
temperature: 17.45000 (float)
wind: 1.95000 (float)

1. add_time_features
--------------------
clouds: 88 (int)
day: 2 (int)
description: overcast clouds (str)
hour: 9 (int)
humidity: 84 (int)
moment: 2016-10-05 09:57:18 (datetime)
pressure: 1,017.34000 (float)
station: pomme (str)
temperature: 17.45000 (float)
wind: 1.95000 (float)

2. Transformer union
--------------------
    2.0 Select
    ----------
    clouds: 88 (int)
    humidity: 84 (int)
    pressure: 1,017.34000 (float)
    temperature: 17.45000 (float)
    wind: 1.95000 (float)

    2.1 TargetAgg
    -------------
    y_mean_by_station_and_hour: 7.89396 (float)

    2.2 TargetAgg1
    --------------
    y_ewm_0.5_by_station: 11.80372 (float)

clouds: 88 (int)
humidity: 84 (int)
pressure: 1,017.34000 (float)
temperature: 17.45000 (float)
wind: 1.95000 (float)
y_ewm_0.5_by_station: 11.80372 (float)
y_mean_by_station_and_hour: 7.89396 (float)

3. StandardScaler
-----------------
clouds: 1.54778 (float)
humidity: 1.16366 (float)
pressure: 0.04916 (float)
temperature: -0.51938 (float)
wind: -0.69426 (float)
y_ewm_0.5_by_station: 0.19640 (float)
y_mean_by_station_and_hour: -0.27110 (float)

4. LinearRegression
-------------------
Name                         Value      Weight     Contribution  
                 Intercept    1.00000    9.19960        9.19960  
      y_ewm_0.5_by_station    0.19640    9.19349        1.80562  
                  humidity    1.16366    1.01680        1.18320  
               temperature   -0.51938   -0.41575        0.21593  
                      wind   -0.69426   -0.03810        0.02645  
                  pressure    0.04916    0.18321        0.00901  
y_mean_by_station_and_hour   -0.27110    0.19553       -0.05301  
                    clouds    1.54778   -0.32838       -0.50827  

Prediction: 11.87854
```

The pipeline does quite a few things, but using `debug_one` shows what happens step by step. This is really useful for checking that the pipeline is behaving as you're expecting it too. Remember that you can `debug_one` whenever you wish, be it before, during, or after training a model.
