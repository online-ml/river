"""
## Introduction

`creme` is a library for in**creme**ntal learning. Incremental learning is a machine learning
regime where the observations are made available one by one. It is also known as online learning,
iterative learning, or sequential learning. This is in contrast to batch learning where all the
data is processed at once. Incremental learning is desirable when the data is too big to fit in
memory, or simply when it isn't available all at once. `creme`'s API is heavily inspired from that
of [scikit-learn](https://scikit-learn.org/stable/), enough so that users who are familiar with
scikit-learn should feel right at home.

Most machine learning algorithms (be it supervised or unsupervised) assume a batch regime. However
some of these algorithms have online variants. For example
[stochastic gradient descent](https://www.wikiwand.com/en/Stochastic_gradient_descent) is the
online version of [gradient descent](https://www.wikiwand.com/en/Gradient_descent) whilst
[incremental k-means clustering](http://www.cs.princeton.edu/courses/archive/fall08/cos436/Duda/C/sk_means.htm)
is the online adaptation of [k-means clustering](https://www.wikiwand.com/en/K-means_clustering).
In general online algorithms perform slightly worse than their batch counterparts, although the gap
is usually very small. However, online learning algorithms only consume a tiny amount of RAM, which
thus makes them scalable and ideal candidates for commodity hardware and embedded systems.

The objective of `creme` is to provide a nice interface for putting an incremental learning
pipeline in place; a bit like what scikit-learn does for batch learning. Of course there are other
open-source solutions available, but they are somewhat specialized towards certain tasks and can
require a steep learning curve. Moreover some of these solutions aren't "truly online" as they
mostly assume the data is contained in a file. With `creme` it is possible to learn from a stream
in it's largest sense, be it a database query or a Kafka instance.

## API

Just like scikit-learn, each of `creme`'s estimators provide a similar API. Every estimator has a
`fit_one(x, y)` method which will fit the estimator with a given set of features `x` and a
target `y`. In addition, estimators have a `predict_one(x)` or a `transform_one(x)` method,
depending on their type. Classifiers also implement a `predict_proba_one(x)` method. Each call to
`fit_one` will also return the predicted value for the current `x`, which makes it possible to
monitor the progress of the estimator online. Although `creme`'s purpose is incremental learning,
each of it's estimators also implements the scikit-learn's `fit/predict/transform` API, which makes
it possible to reuse scikit-learn's toolbox. This is mostly intended to help comparing batch
algorithms to their online versions.

Rows in `creme` are represented by `dict`s that map feature names to values. The main advantage of
using a `dict` over a `numpy` array is that features can be accessed by name rather by position.
Moreover `dict`s can store values of different types, whereas all the values in a `numpy` array
have to be of a single type. What's more `dict`s are sparse by default; indeed a feature with a
`None` value can simply be omitted. Finally Python has some nice tools for working with `dict`s,
such as `collections.defaultdict`.

The `creme` library is organized in modules, following the fashion in which scikit-learn is
organized. There is thus a `preprocessing` module as well as a `feature_extraction` module, amongst
others. Furthermore an effort is made to keep the naming of `creme`'s classes and parameters
consistent with scikit-learn.

For more information please check out the [documentation](https://creme.github.io).

## Important links

- Documentation
- Example notebooks
- Issue tracker
- Releases

## Installation

```sh
pip install creme
```

## Quick example

In the following snippet we'll be fitting an online logistic regression. The weights of the model
will be optimized with the [AdaGrad](http://akyrillidis.github.io/notes/AdaGrad) algorithm. We'll
scale the data so that each variable has a mean of 0 and a standard deviation of 1. The standard
scaling and the logistic regression are combined using a pipeline. We'll be using the
`stream.iter_sklearn_dataset` function for streaming over the
[Wisconsin breast cancer dataset](http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29).
We'll measure the ROC AUC using [progressive validation](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.153.3925&rep=rep1&type=pdf).

```python
>>> from creme import linear_model
>>> from creme import model_selection
>>> from creme import optim
>>> from creme import pipeline
>>> from creme import preprocessing
>>> from creme import stream
>>> from sklearn import datasets
>>> from sklearn import metrics

>>> X_y = stream.iter_sklearn_dataset(
...     load_dataset=datasets.load_breast_cancer,
...     shuffle=True,
...     random_state=42
... )
>>> optimizer = optim.AdaGrad()
>>> model = pipeline.Pipeline([
...     ('scale', preprocessing.StandardScaler()),
...     ('learn', linear_model.LogisticRegression(optimizer))
... ])
>>> metric = metrics.roc_auc_score

>>> model_selection.online_score(X_y, model, metric)
0.992977...
```

"""
from .__version__ import __version__

__all__ = [
    'cluster',
    'compose',
    'ensemble',
    'feature_extraction',
    'linear_model',
    'losses',
    'model_selection',
    'naive_bayes',
    'optim',
    'pipeline',
    'preprocessing',
    'reco',
    'stats',
    'stream',
    'tree'
]
