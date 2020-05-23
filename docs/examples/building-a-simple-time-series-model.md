# Building a simple time series model


```python
%matplotlib inline
```

We'll be using the international airline passenger data available from [here](https://datamarket.com/data/set/22u3/international-airline-passengers-monthly-totals-in-thousands-jan-49-dec-60#!ds=22u3&display=line). This particular dataset is included with `creme` in the `datasets` module.


```python
from creme import datasets

for x, y in datasets.Airline():
    print(x, y)
    break
```

    {'month': datetime.datetime(1949, 1, 1, 0, 0)} 112


The data is as simple as can be: it consists of a sequence of months and values representing the total number of international airline passengers per month. Our goal is going to be to predict the number of passengers for the next month at each step. Notice that because the dataset is small  -- which is usually the case for time series -- we could just fit a model from scratch each month. However for the sake of example we're going to train a single model online. Although the overall performance might be potentially weaker, training a time series model online has the benefit of being scalable if, say, you have have [thousands of time series to manage](http://www.unofficialgoogledatascience.com/2017/04/our-quest-for-robust-time-series.html).

We'll start with a very simple model where the only feature will be the [ordinal date](https://www.wikiwand.com/en/Ordinal_date) of each month. This should be able to capture some of the underlying trend. 


```python
from creme import compose
from creme import linear_model
from creme import preprocessing


def get_ordinal_date(x):
    return {'ordinal_date': x['month'].toordinal()}


model = compose.Pipeline(
    ('ordinal_date', compose.FuncTransformer(get_ordinal_date)),
    ('scale', preprocessing.StandardScaler()),
    ('lin_reg', linear_model.LinearRegression())
)
```

We'll write down a function to evaluate the model. This will go through each observation in the dataset and update the model as it goes on. The prior predictions will be stored along with the true values and will be plotted together. 


```python
from creme import metrics
import matplotlib.pyplot as plt


def evaluate_model(model): 
    
    metric = metrics.Rolling(metrics.MAE(), 12)
    
    dates = []
    y_trues = []
    y_preds = []

    for x, y in datasets.Airline():
        
        # Obtain the prior prediction and update the model in one go
        y_pred = model.predict_one(x)
        model.fit_one(x, y)
        
        # Update the error metric
        metric.update(y, y_pred)
        
        # Store the true value and the prediction
        dates.append(x['month'])
        y_trues.append(y)
        y_preds.append(y_pred)
        
    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(alpha=0.75)
    ax.plot(dates, y_trues, lw=3, color='#2ecc71', alpha=0.8, label='Truth')
    ax.plot(dates, y_preds, lw=3, color='#e74c3c', alpha=0.8, label='Prediction')
    ax.legend()
    ax.set_title(metric)
```

Let's evaluate our first model.


```python
evaluate_model(model)
```


![png](building-a-simple-time-series-model_files/building-a-simple-time-series-model_9_0.png)


The model has captured a trend but not the right one. Indeed it thinks the trend is linear whereas we can visually see that the growth of the data increases with time. In other words the second derivative of the series is positive. This is a well know problem in time series forecasting and there are thus many ways to handle it; for example by using a [Box-Cox transform](https://www.wikiwand.com/en/Power_transform). However we are going to do something a bit different, and instead linearly detrend the series using a `Detrender`. We'll set `window_size` to 12 in order to use a rolling mean of size 12 for detrending. The `Detrender` will center the target in 0, which means that we don't need an intercept in our linear regression. We can thus set `intercept_lr` to 0.


```python
from creme import stats
from creme import time_series


model = compose.Pipeline(
    ('ordinal_date', compose.FuncTransformer(get_ordinal_date)),
    ('scale', preprocessing.StandardScaler()),
    ('lin_reg', linear_model.LinearRegression(intercept_lr=0)),
)

model = time_series.Detrender(regressor=model, window_size=12)

evaluate_model(model)
```


![png](building-a-simple-time-series-model_files/building-a-simple-time-series-model_11_0.png)


Now let's try and capture the monthly trend by one-hot encoding the month name.


```python
import calendar


def get_month(x):
    return {
        calendar.month_name[month]: month == x['month'].month
        for month in range(1, 13)
    }
    

model = compose.Pipeline(
    ('features', compose.TransformerUnion(
        ('ordinal_date', compose.FuncTransformer(get_ordinal_date)),
        ('month', compose.FuncTransformer(get_month)),
    )),
    ('scale', preprocessing.StandardScaler()),
    ('lin_reg', linear_model.LinearRegression(intercept_lr=0))
)

model = time_series.Detrender(regressor=model, window_size=12)

evaluate_model(model)
```


![png](building-a-simple-time-series-model_files/building-a-simple-time-series-model_13_0.png)


This seems pretty decent. We can take a look at the weights of the linear regression to get an idea of the importance of each feature.


```python
model.regressor['lin_reg'].weights
```




    defaultdict(<creme.optim.initializers.Zeros at 0x151b4b7f50>,
                {'January': -5.22573277384658,
                 'February': -7.296050401976386,
                 'March': -0.6880089328552284,
                 'April': -1.4296293828257067,
                 'May': -0.7567340334065892,
                 'June': 6.906892452281285,
                 'July': 14.593482574659191,
                 'August': 13.825902468738116,
                 'September': 2.9030216363458914,
                 'October': -4.863645430335497,
                 'November': -12.28084856162094,
                 'December': -6.875234471703001,
                 'ordinal_date': 11.033605877732066})



As could be expected the months of July and August have the highest weights because these are the months where people typically go on holiday abroad. The month of December has a low weight because this is a month of festivities in most of the Western world where people usually stay at home.

Our model seems to understand which months are important, but it fails to see that the importance of each month grows multiplicatively as the years go on. In other words our model is too shy. We can fix this by increasing the learning rate of the `LinearRegression`'s optimizer.


```python
from creme import optim

model = compose.Pipeline(
    ('features', compose.TransformerUnion(
        ('ordinal_date', compose.FuncTransformer(get_ordinal_date)),
        ('month', compose.FuncTransformer(get_month)),
    )),
    ('scale', preprocessing.StandardScaler()),
    ('lin_reg', linear_model.LinearRegression(
        intercept_lr=0,
        optimizer=optim.SGD(0.03)
    ))
)

model = time_series.Detrender(regressor=model, window_size=12)

evaluate_model(model)
```


![png](building-a-simple-time-series-model_files/building-a-simple-time-series-model_17_0.png)


This is starting to look good! Naturally in production we would tune the learning rate, ideally in real-time.

Before finishing, we're going to introduce a cool feature extraction trick based on [radial basis function kernels](https://www.wikiwand.com/en/Radial_basis_function_kernel). The one-hot encoding we did on the month is a good idea but if you think about it is a bit rigid. Indeed the value of each feature is going to be 0 or 1, depending on the month of each observation. We're basically saying that the month of September is as distant to the month of August as it is to the month of March. Of course this isn't true, and it would be nice if our features would reflect this. To do so we can simply calculate the distance between the month of each observation and all the months in the calendar. Instead of simply computing the distance linearly, we're going to use a so-called *Gaussian radial basic function kernel*. This is a bit of a mouthful but for us it boils down to a simple formula, which is:

$$d(i, j) = exp(-\frac{(i - j)^2}{2\sigma^2})$$

Intuitively this computes a similarity between two months -- denoted by $i$ and $j$ -- which decreases the further apart they are from each other. The $sigma$ parameter can be seen as a hyperparameter than can be tuned -- in the following snippet we'll simply ignore it. The thing to take away is that this results in smoother predictions than when using a one-hot encoding scheme, which is often a desirable property. You can also see trick in action [in this nice presentation](http://www.youtube.com/watch?v=68ABAU_V8qI&t=4m45s).


```python
import math

def get_month_distances(x):
    return {
        calendar.month_name[month]: math.exp(-(x['month'].month - month) ** 2)
        for month in range(1, 13)
    }
    

model = compose.Pipeline(
    ('features', compose.TransformerUnion(
        ('ordinal_date', compose.FuncTransformer(get_ordinal_date)),
        ('month_distances', compose.FuncTransformer(get_month_distances)),
    )),
    ('scale', preprocessing.StandardScaler()),
    ('lin_reg', linear_model.LinearRegression(
        intercept_lr=0,
        optimizer=optim.SGD(0.03)
    ))
)

model = time_series.Detrender(regressor=model, window_size=12)

evaluate_model(model)
```


![png](building-a-simple-time-series-model_files/building-a-simple-time-series-model_19_0.png)


We've managed to get a good looking prediction curve with a reasonably simple model. What's more our model has the advantage of being interpretable and easy to debug. There surely are more rocks to squeeze (e.g. tune the hyperparameters, use an ensemble model, etc.) but we'll leave that as an exercice to the reader.

As a finishing touch we'll rewrite our pipeline using the `|` operator, which is called a "pipe".


```python
extract_features = compose.TransformerUnion(get_ordinal_date, get_month_distances)

scale = preprocessing.StandardScaler()

learn = linear_model.LinearRegression(
    intercept_lr=0,
    optimizer=optim.SGD(0.03)
)

model = extract_features | scale | learn
model = time_series.Detrender(regressor=model, window_size=12)

evaluate_model(model)
```


![png](building-a-simple-time-series-model_files/building-a-simple-time-series-model_21_0.png)



```python

```
