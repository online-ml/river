# Bike-sharing forecasting (regression)

In this tutorial we're going to forecast the number of bikes in 5 bike stations from the city of Toulouse. We'll do so by building a simple model step by step. The dataset contains 182,470 observations. Let's first take a peak at the data.


```python
from pprint import pprint
from creme import datasets

X_y = datasets.Bikes()

for x, y in X_y:
    pprint(x)
    print(f'Number of available bikes: {y}')
    break
```

    {'clouds': 75,
     'description': 'light rain',
     'humidity': 81,
     'moment': datetime.datetime(2016, 4, 1, 0, 0, 7),
     'pressure': 1017.0,
     'station': 'metro-canal-du-midi',
     'temperature': 6.54,
     'wind': 9.3}
    Number of available bikes: 1


Let's start by using a simple linear regression on the numeric features. We can select the numeric features and discard the rest of the features using a `Select`. Linear regression is very likely to go haywire if we don't scale the data, so we'll use a `StandardScaler` to do just that. We'll evaluate the model by measuring the mean absolute error. Finally we'll print the score every 20,000 observations. 


```python
from creme import compose
from creme import linear_model
from creme import metrics
from creme import model_selection
from creme import preprocessing

X_y = datasets.Bikes()

model = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
model |= preprocessing.StandardScaler()
model |= linear_model.LinearRegression()

metric = metrics.MAE()

model_selection.progressive_val_score(X_y, model, metric, print_every=20_000)
```

    [20,000] MAE: 4.914777
    [40,000] MAE: 5.336598
    [60,000] MAE: 5.337153
    [80,000] MAE: 5.403989
    [100,000] MAE: 5.439635
    [120,000] MAE: 5.55724
    [140,000] MAE: 5.629535
    [160,000] MAE: 5.640354
    [180,000] MAE: 5.590733





    MAE: 5.587171



The model doesn't seem to be doing that well, but then again we didn't provide with a lot of features. A generally good idea for this kind of problem is to look at an average of the previous values. For example, for each station we can look at the average number of bikes per hour. To do so we first have to extract the hour from the  `moment` field. We can then use a `TargetAgg` to aggregate the values of the target.


```python
from creme import feature_extraction
from creme import stats

X_y = iter(datasets.Bikes())

def get_hour(x):
    x['hour'] = x['moment'].hour
    return x

model = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
model += (
    get_hour |
    feature_extraction.TargetAgg(by=['station', 'hour'], how=stats.Mean())
)
model |= preprocessing.StandardScaler()
model |= linear_model.LinearRegression()

metric = metrics.MAE()

model_selection.progressive_val_score(X_y, model, metric, print_every=20_000)
```

    [20,000] MAE: 3.692385
    [40,000] MAE: 3.833377
    [60,000] MAE: 3.861924
    [80,000] MAE: 3.932616
    [100,000] MAE: 3.914911
    [120,000] MAE: 3.95326
    [140,000] MAE: 4.014267
    [160,000] MAE: 3.987742
    [180,000] MAE: 3.979413





    MAE: 3.977571



By adding a single feature, we've managed to significantly reduce the mean absolute error. At this point you might think that the model is getting slightly complex, and is difficult to understand and test. Pipelines have the advantage of being terse, but they aren't always to debug. Thankfully `creme` has some ways to relieve the pain.

The first thing we can do it to draw the pipeline, to get an idea of how the data flows through it.


```python
model.draw()
```




![svg](bike-sharing-forecasting_files/bike-sharing-forecasting_8_0.svg)



We can also use the `debug_one` method to see what happens to one particular instance. Let's train the model on the first 10,000 observations and then call `debug_one` on the next one. To do this, we will turn the `Bike` object into a Python generator with `iter()` function. The Pythonic way to read the first 10,000 elements of a generator is to use `itertools.islice`.


```python
import itertools

X_y = iter(datasets.Bikes())

model = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
model += (
    get_hour |
    feature_extraction.TargetAgg(by=['station', 'hour'], how=stats.Mean())
)
model |= preprocessing.StandardScaler()
model |= linear_model.LinearRegression()

for x, y in itertools.islice(X_y, 10000):
    y_pred = model.predict_one(x)
    model.fit_one(x, y)
    
x, y = next(X_y)
model.debug_one(x)
```

    0. Input
    --------
    clouds: 0 (int)
    description: clear sky (str)
    humidity: 52 (int)
    moment: 2016-04-10 19:03:27 (datetime)
    pressure: 1,001.00000 (float)
    station: place-esquirol (str)
    temperature: 19.00000 (float)
    wind: 7.70000 (float)
    
    1. Transformer union
    --------------------
        1.0 ('clouds', 'humidity', 'pressure', 'temperature', 'wind')
        -------------------------------------------------------------
        clouds: 0 (int)
        humidity: 52 (int)
        pressure: 1,001.00000 (float)
        temperature: 19.00000 (float)
        wind: 7.70000 (float)
    
        1.1 get_hour | target_mean_by_station_and_hour
        ----------------------------------------------
        target_mean_by_station_and_hour: 7.97175 (float)
    
    clouds: 0 (int)
    humidity: 52 (int)
    pressure: 1,001.00000 (float)
    target_mean_by_station_and_hour: 7.97175 (float)
    temperature: 19.00000 (float)
    wind: 7.70000 (float)
    
    2. StandardScaler
    -----------------
    clouds: -1.36131 (float)
    humidity: -1.73074 (float)
    pressure: -1.26070 (float)
    target_mean_by_station_and_hour: 0.05495 (float)
    temperature: 1.76223 (float)
    wind: 1.45834 (float)
    
    3. LinearRegression
    -------------------
    Name                              Value      Weight     Contribution  
                          Intercept    1.00000    7.43988        7.43988  
                        temperature    1.76223    0.43250        0.76216  
    target_mean_by_station_and_hour    0.05495    2.30036        0.12641  
                           humidity   -1.73074   -0.04168        0.07214  
                             clouds   -1.36131    0.19230       -0.26177  
                               wind    1.45834   -0.44409       -0.64763  
                           pressure   -1.26070    1.65184       -2.08246  
    
    Prediction: 5.40872


The `debug_one` method shows what happens to an input set of features, step by step.

And now comes the catch. Up until now we've been using the `online_score` method from the `model_selection` module. What this does it that it sequentially predicts the output of an observation and updates the model immediately afterwards. This way of doing is often used for evaluating online learning models, but in some cases it is the wrong approach. 

The following paragraph is extremely important. When evaluating a machine learning model, the goal is to simulate production conditions in order to get a trust-worthy assessment of the performance of the model. In our case, we typically want to forecast the number of bikes available in a station, say, 30 minutes ahead. Then, once the 30 minutes have passed, the true number of available bikes will be available and we will be able to update the model using the features available 30 minutes ago. If you think about, this is exactly how a real-time machine learning system should work. The problem is that this isn't at all what the `online_score` method, indeed it is simply asking the model to predict the next observation, which is only a few minutes ahead, and then updates the model immediately. We can prove that this is flawed by adding a feature that measures a running average of the very recent values.


```python
X_y = datasets.Bikes()

model = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
model += (
    get_hour |
    feature_extraction.TargetAgg(by=['station', 'hour'], how=stats.Mean()) + 
    feature_extraction.TargetAgg(by='station', how=stats.EWMean(0.5))
)
model |= preprocessing.StandardScaler()
model |= linear_model.LinearRegression()

metric = metrics.MAE()

model_selection.progressive_val_score(X_y, model, metric, print_every=20_000)
```

    [20,000] MAE: 0.796592
    [40,000] MAE: 0.773377
    [60,000] MAE: 0.809594
    [80,000] MAE: 0.859641
    [100,000] MAE: 0.886466
    [120,000] MAE: 0.896317
    [140,000] MAE: 0.901659
    [160,000] MAE: 0.914834
    [180,000] MAE: 0.924302





    MAE: 0.92614



The score we got is too good to be true. This is simply because the problem is too easy. What we really want is to evaluate the model by forecasting 30 minutes ahead and only updating the model once the true values are available. This can be done using the `online_qa_score` method, also from the `model_selection` module. The "qa" part stands for "question/answer". The idea is that each observation of the stream of the data is shown twice to the model: once for making a prediction, and once for updating the model when the true value is revealed. The `on` parameter determines which variable should be used as a timestamp, while the `lag` parameter controls the duration to wait before revealing the true values to the model.


```python
import datetime as dt

model = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
model += (
    get_hour |
    feature_extraction.TargetAgg(by=['station', 'hour'], how=stats.Mean()) + 
    feature_extraction.TargetAgg(by='station', how=stats.EWMean(0.5))
)
model |= preprocessing.StandardScaler()
model |= linear_model.LinearRegression()

model_selection.progressive_val_score(
    X_y=datasets.Bikes(),
    model=model,
    metric=metrics.MAE(),
    moment='moment',
    delay=dt.timedelta(minutes=30),
    print_every=20_000
)
```

    [20,000] MAE: 2.209483
    [40,000] MAE: 2.216244
    [60,000] MAE: 2.249483
    [80,000] MAE: 2.271607
    [100,000] MAE: 2.281669
    [120,000] MAE: 2.26541
    [140,000] MAE: 2.252277
    [160,000] MAE: 2.27968
    [180,000] MAE: 2.282643





    MAE: 2.286077



The score we now have is much more realistic, as it is comparable with [related data science competitions](https://maxhalford.github.io/blog/a-short-introduction-and-conclusion-to-the-openbikes-2016-challenge/). Moreover, we can see that the model gets better with time, which feels better than the previous situations. The point is that `online_qa_score` method can be used to simulate a production scenario, and is thus extremely valuable.

Now that we have a working pipeline in place, we can attempt to make it more accurate. As a simple example, we'll using a `HedgeRegressor` from the `ensemble` module to combine 3 linear regression model trained with different optimizers. The `HedgeRegressor` will run the 3 models in parallel and assign weights to each model based on their individual performance.


```python
from creme import ensemble
from creme import optim

model = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
model += (
    get_hour |
    feature_extraction.TargetAgg(by=['station', 'hour'], how=stats.Mean())
)
model += feature_extraction.TargetAgg(by='station', how=stats.EWMean(0.5))
model |= preprocessing.StandardScaler()
model |= ensemble.HedgeRegressor([
    linear_model.LinearRegression(optim.SGD()),
    linear_model.LinearRegression(optim.RMSProp()),
    linear_model.LinearRegression(optim.Adam())
])

model_selection.progressive_val_score(
    X_y=datasets.Bikes(),
    model=model,
    metric=metrics.MAE(),
    moment='moment',
    delay=dt.timedelta(minutes=30),
    print_every=20_000
)
```

    [20,000] MAE: 2.252008
    [40,000] MAE: 2.24223
    [60,000] MAE: 2.271582
    [80,000] MAE: 2.287461
    [100,000] MAE: 2.295041
    [120,000] MAE: 2.276539
    [140,000] MAE: 2.261966
    [160,000] MAE: 2.286464
    [180,000] MAE: 2.289785





    MAE: 2.293466


