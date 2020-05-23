```python
%load_ext autoreload
%autoreload 2
```

# Introduction

## A quick overview of batch learning

If you've already delved into machine learning, then you shouldn't have any difficulty in getting to use incremental learning. If you are somewhat new to machine learning, then do not worry! The point of this notebook in particular is to introduce simple notions. We'll also start to show how `creme` fits in and explain how to use it.

The whole point of machine learning is to *learn from data*. In *supervised learning* you want to learn how to predict a target $y$ given a set of features $X$. Meanwhile in an unsupervised learning there is no target, and the goal is rather to identify patterns and trends in the features $X$. At this point most people tend to imagine $X$ as a somewhat big table where each row is an observation and each column is a feature, and they would be quite right. Learning from tabular data is part of what's called *batch learning*, which basically that all of the data is available to our learning algorithm at once. A lot of libraries have been created to handle the batch learning regime, with one of the most prominent being Python's [scikit-learn](https://scikit-learn.org/stable/). 

As a simple example of batch learning let's say we want to learn to predict if a women has breast cancer or not. We'll use the [breast cancer dataset available with scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer().html). We'll learn to map a set of features to a binary decision using a [logistic regression](https://www.wikiwand.com/en/Logistic_regression). Like many other models based on numerical weights, logisitc regression is sensitive to the scale of the features. Rescaling the data so that each feature has mean 0 and variance 1 is generally considered good practice. We can apply the rescaling and fit the logistic regression sequentially in an elegant manner using a [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html). To measure the performance of the model we'll evaluate the average [ROC AUC score](https://www.wikiwand.com/en/Receiver_operating_characteristic) using a 5 fold [cross-validation](https://www.wikiwand.com/en/Cross-validation_(statistics)). 


```python
from sklearn import datasets
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn import pipeline
from sklearn import preprocessing


# Load the data
dataset = datasets.load_breast_cancer()
X, y = dataset.data, dataset.target

# Define the steps of the model
model = pipeline.Pipeline([
    ('scale', preprocessing.StandardScaler()),
    ('lin_reg', linear_model.LogisticRegression(solver='lbfgs'))
])

# Define a determistic cross-validation procedure
cv = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)

# Compute the MSE values
scorer = metrics.make_scorer(metrics.roc_auc_score)
scores = model_selection.cross_val_score(model, X, y, scoring=scorer, cv=cv)

# Display the average score and it's standard deviation
print(f'ROC AUC: {scores.mean():.3f} (± {scores.std():.3f})')
```

    ROC AUC: 0.975 (± 0.011)


This might be a lot to take in if you're not accustomed to scikit-learn, but it probably isn't if you are. Batch learning basically boils down to:

1. Loading the data
2. Fitting a model to the data
3. Computing the performance of the model on unseen data

This is pretty standard and is maybe how most people imagine a machine learning pipeline. However this way of proceding has certain downsides. First of all your laptop would crash if the `load_boston` function returned a dataset who's size exceeds your available amount of RAM. Sometimes you can use some tricks to get around this. For example by optimizing the data types and by using sparse representations when applicable you can potentially save precious gigabytes of RAM. However like many tricks this only goes so far. If your dataset weighs hundreds of gigabytes then you won't go far without some special hardware. One solution is to do out-of-core learning; that is, algorithms that can learning by being presented the data in chunks. If you want to go down this road then take a look at [Dask](https://examples.dask.org/machine-learning.html) and [Spark's MLlib](https://spark.apache.org/mllib/).

Another issue with the batch learning regime is that can't elegantly learn from new data. Indeed if new data is made available, then the model has to learn from scratch with a new dataset composed of the old data and the new data. This is particularly annoying in a real situation where you might have new incoming data every week, day, hour, minute, or even setting. For example if you're building a recommendation engine for an e-commerce app, then you're probably training your model from 0 every week or so. As your app grows in popularity, so does the dataset you're training on. This will lead to longer and longer training times and might require a hardware upgrade.

A final downside that isn't very easy to grasp concerns the manner in which features are extracted. Everytime you want to train your model you first have to extract features. The trick is that some features might not be accessible at the particular point in time you are at. For example maybe that some attributes in your data warehouse get overwritten with time. In other words maybe that all the features pertaining to a particular observations are not available, whereas they were a week ago. This happens more often than not in real scenarios, and apart if you have a sophisticated data engineering pipeline then you will encounter these issues at some point. 

## A hands-on introduction to incremental learning

Incremental learning is also often called *online learning*, but if you [google online learning](https://www.google.com/search?q=online+learning) a lot of the results will point to educational websites. Hence we prefer the name "incremental learning", from which `creme` derives it's name. The point of incremental learning is to fit a model to a stream of data. In other words, the data isn't available in it's entirety, but rather the observations are provided one by one. As an example let's stream through the dataset used previously.


```python
for xi, yi in zip(X, y):
    # This where the model learns
    pass
```

In this case we're iterating over a dataset that is already in memory, but we could just as well stream from a CSV file, a Kafka stream, an SQL query, etc. If we look at `x` we can notice that it is a `numpy.ndarray`.


```python
xi
```




    array([7.760e+00, 2.454e+01, 4.792e+01, 1.810e+02, 5.263e-02, 4.362e-02,
           0.000e+00, 0.000e+00, 1.587e-01, 5.884e-02, 3.857e-01, 1.428e+00,
           2.548e+00, 1.915e+01, 7.189e-03, 4.660e-03, 0.000e+00, 0.000e+00,
           2.676e-02, 2.783e-03, 9.456e+00, 3.037e+01, 5.916e+01, 2.686e+02,
           8.996e-02, 6.444e-02, 0.000e+00, 0.000e+00, 2.871e-01, 7.039e-02])



`creme` on the other hand works with `dict`s. We believe that `dict`s are more enjoyable to program with than `numpy.ndarray`s, at least for when single observations are concerned. `dict`'s bring the added benefit that each feature can be accessed by name rather than by position.


```python
for xi, yi in zip(X, y):
    xi = dict(zip(dataset.feature_names, xi))
    pass

xi
```




    {'mean radius': 7.76,
     'mean texture': 24.54,
     'mean perimeter': 47.92,
     'mean area': 181.0,
     'mean smoothness': 0.05263,
     'mean compactness': 0.04362,
     'mean concavity': 0.0,
     'mean concave points': 0.0,
     'mean symmetry': 0.1587,
     'mean fractal dimension': 0.05884,
     'radius error': 0.3857,
     'texture error': 1.428,
     'perimeter error': 2.548,
     'area error': 19.15,
     'smoothness error': 0.007189,
     'compactness error': 0.00466,
     'concavity error': 0.0,
     'concave points error': 0.0,
     'symmetry error': 0.02676,
     'fractal dimension error': 0.002783,
     'worst radius': 9.456,
     'worst texture': 30.37,
     'worst perimeter': 59.16,
     'worst area': 268.6,
     'worst smoothness': 0.08996,
     'worst compactness': 0.06444,
     'worst concavity': 0.0,
     'worst concave points': 0.0,
     'worst symmetry': 0.2871,
     'worst fractal dimension': 0.07039}



`creme`'s `stream` module has an `iter_sklearn_dataset` convenience function that we can use instead.


```python
from creme import stream

for xi, yi in stream.iter_sklearn_dataset(datasets.load_breast_cancer()):
    pass
```

The simple fact that we are getting the data in a stream means that we can't do a lot of things the same way as in a batch setting. For example let's say we want to scale the data so that it has mean 0 and variance 1, as we did earlier. To do so we simply have to subtract the mean of each feature to each value and then divide the result by the standard deviation of the feature. The problem is that we can't possible known the values of the mean and the standard deviation before actually going through all the data! One way to procede would be to do a first pass over the data to compute the necessary values and then scale the values during a second pass. The problem is that defeats our purpose, which is to learn by only looking at the data once. Although this might seem rather restrictive, it reaps sizable benefits down the road.

The way we do feature scaling in `creme` involves computing *running statistics*. The idea is that we use a data structure that estimates the mean and updates itself when it is provided with a value. The same goes for the variance (and thus the standard deviation). For example, if we denote $\mu_t$ the mean and $n_t$ the count at any moment $t$, then updating the mean can be done as so:

$$
\begin{cases}
n_{t+1} = n_t + 1 \\
\mu_{t+1} = \mu_t + \frac{x - \mu_t}{n_{t+1}}
\end{cases}
$$

Likewhise a running variance can be computed as so:

$$
\begin{cases}
n_{t+1} = n_t + 1 \\
\mu_{t+1} = \mu_t + \frac{x - \mu_t}{n_{t+1}} \\
s_{t+1} = s_t + (x - \mu_t) \times (x - \mu_{t+1}) \\
\sigma_{t+1} = \frac{s_{t+1}}{n_{t+1}}
\end{cases}
$$

where $s_t$ is a running sum of squares and $\sigma_t$ is the running variance at time $t$. This might seem a tad more involved than the batch algorithms you learn in school, but it is rather elegant. Implementing this in Python is not too difficult. For example let's compute the running mean and variance of the `'mean area'` variable.


```python
n, mean, sum_of_squares, variance = 0, 0, 0, 0

for xi, yi in stream.iter_sklearn_dataset(datasets.load_breast_cancer()):
    n += 1
    old_mean = mean
    mean += (xi['mean area'] - mean) / n
    sum_of_squares += (xi['mean area'] - old_mean) * (xi['mean area'] - mean)
    variance = sum_of_squares / n
    
print(f'Running mean: {mean:.3f}')
print(f'Running variance: {variance:.3f}')
```

    Running mean: 654.889
    Running variance: 123625.903


Let's compare this with `numpy`.


```python
import numpy as np

i = list(dataset.feature_names).index('mean area')
print(f'True mean: {np.mean(X[:, i]):.3f}')
print(f'True variance: {np.var(X[:, i]):.3f}')
```

    True mean: 654.889
    True variance: 123625.903


The results seem to be exactly the same! The twist is that the running statistics won't be very accurate for the first few observations. In general though this doesn't matter too much. Some would even go as far as to say that this descrepancy is beneficial and acts as some sort of regularization...

Now the idea is that we can compute the running statistics of each feature and scale them as they come along. The way to do this with `creme` is to use the `StandardScaler` class from the `preprocessing` module, as so:


```python
from creme import preprocessing

scaler = preprocessing.StandardScaler()

for xi, yi in stream.iter_sklearn_dataset(datasets.load_breast_cancer()):
    xi = scaler.fit_one(xi, yi)
```

This is quite terse but let's break it down nonetheless. Every class in `creme` has a `fit_one(x, y)` method where all the magic happens. Now the important thing to notice is that the `fit_one` actually returns the output for the given input. This is one of the nice properties of online learning: inference can be done immediatly. In `creme` each call to a `Transformer`'s `fit_one` will return the transformed output. Meanwhile calling `fit_one` with a `Classifier` or a `Regressor` will return the predicted target for the given set of features. The twist is that the prediction is made *before* looking at the true target `y`. This means that we get a free hold-out prediction every time we call `fit_one`. This can be used to monitor the performance of the model as it trains, which is obviously nice to have.

Now that we are scaling the data, we can start doing some actual machine learning. We're going to implement an online linear regression. Because all the data isn't available at once, we are obliged to do what is called *stochastic gradient descent*, which is a popular research topic and has a lot of variants. SGD is commonly used to train neural networks. The idea is that at each step we compute the loss between the target prediction and the truth. We then calculate the gradient, which is simply a set of derivatives with respect to each weight from the linear regression. Once we have obtained the gradient, we can update the weights by moving them in the opposite direction of the gradient. The amount by which the weights are moved typically depends on a *learning rate*, which is typically set by the user. Different optimizers have different ways of managing the weight update, and some handle the learning rate implicitely. Online linear regression can be done in `creme` with the `LinearRegression` class from the `linear_model` module. We'll be using plain and simple SGD using the `SGD` optimizer from the `optim` module. During training we'll measure the squared error between the truth and the predictions.


```python
from creme import linear_model
from creme import optim

scaler = preprocessing.StandardScaler()
optimizer = optim.SGD(lr=0.01)
log_reg = linear_model.LogisticRegression(optimizer)

y_true = []
y_pred = []

for xi, yi in stream.iter_sklearn_dataset(datasets.load_breast_cancer(), shuffle=True, seed=42):
    
    # Scale the features
    xi_scaled = scaler.fit_one(xi).transform_one(xi)
    
    # Fit the linear regression
    yi_pred = log_reg.predict_proba_one(xi_scaled)
    log_reg.fit_one(xi_scaled, yi)
    
    # Store the truth and the prediction
    y_true.append(yi)
    y_pred.append(yi_pred[True])
    
print(f'ROC AUC: {metrics.roc_auc_score(y_true, y_pred):.3f}')
```

    ROC AUC: 0.989


The ROC AUC is significantly better than the one obtained from the cross-validation of scikit-learn's logisitic regression. However to make things really comparable it would be nice to compare with the same cross-validation procedure. `creme` has a `compat` module that contains utilities for making `creme` compatible with other Python libraries. Because we're doing regression we'll be using the `SKLRegressorWrapper`. We'll also be using `Pipeline` to encapsulate the logic of the `StandardScaler` and the `LogisticRegression` in one single object.


```python
from creme import compat
from creme import compose

# We define a Pipeline, exactly like we did earlier for sklearn 
model = compose.Pipeline(
    ('scale', preprocessing.StandardScaler()),
    ('log_reg', linear_model.LogisticRegression())
)

# We make the Pipeline compatible with sklearn
model = compat.convert_creme_to_sklearn(model)

# We compute the CV scores using the same CV scheme and the same scoring
scores = model_selection.cross_val_score(model, X, y, scoring=scorer, cv=cv)

# Display the average score and it's standard deviation
print(f'ROC AUC: {scores.mean():.3f} (± {scores.std():.3f})')
```

    ROC AUC: 0.964 (± 0.016)


This time the ROC AUC score is lower, which is what we would expect. Indeed online learning isn't as accurate as batch learning. However it all depends in what you're interested in. If you're only interested in predicting the next observation then the online learning regime would be better. That's why it's a bit hard to compare both approaches: they're both suited to different scenarios.

## Going further

There's a lot more to learn, and it all depends on what kind on your use case. Feel free to have a look at the [documentation](https://creme-ml.github.io/) to know what `creme` has available, and have a look the [example notebook](https://github.com/creme-ml/notebooks).

Here a few resources if you want to do some reading:

- [Online learning -- Wikipedia](https://www.wikiwand.com/en/Online_machine_learning)
- [What is online machine learning? -- Max Pagels](https://medium.com/value-stream-design/online-machine-learning-515556ff72c5)
- [Introduction to Online Learning -- USC course](http://www-bcf.usc.edu/~haipengl/courses/CSCI699/)
- [Online Methods in Machine Learning -- MIT course](http://www.mit.edu/~rakhlin/6.883/)
- [Online Learning: A Comprehensive Survey](https://arxiv.org/pdf/1802.02871.pdf)
- [Streaming 101: The world beyond batch](https://www.oreilly.com/ideas/the-world-beyond-batch-streaming-101)
- [Machine learning for data streams](https://www.cms.waikato.ac.nz/~abifet/book/contents.html)
- [Data Stream Mining: A Practical Approach](https://www.cs.waikato.ac.nz/~abifet/MOA/StreamMining.pdf)

