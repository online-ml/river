# Mini-batching

In its purest form, online machine learning encompasses models which learn with one sample at a time. This is the design which is used in `river`.

The main downside of single-instance processing is that it doesn't scale to big data, at least not in the sense of traditional batch learning. Indeed, processing one sample at a time means that we are unable to fully take advantage of [vectorisation](https://www.wikiwand.com/en/Vectorization) and other computational tools that are taken for granted in batch learning. On top of this, processing a large dataset in `river` essentially involves a Python `for` loop, which might be too slow for some usecases. However, this doesn't mean that `river` is slow. In fact, for processing a single instance, `river` is actually a couple of orders of magnitude faster than libraries such as scikit-learn, PyTorch, and Tensorflow. The reason why is because `river` is designed from the ground up to process a single instance, whereas the majority of other libraries choose to care about batches of data. Both approaches offer different compromises, and the best choice depends on your usecase.

In order to propose the best of both worlds, `river` offers some limited support for mini-batch learning. Some of `river`'s estimators implement `*_many` methods on top of their `*_one` counterparts. For instance, `preprocessing.StandardScaler` has a `learn_many` method as well as a `transform_many` method, in addition to `learn_one` and `transform_one`. Each mini-batch method takes as input a `pandas.DataFrame`. Supervised estimators also take as input a `pandas.Series` of target values. We choose to use `pandas.DataFrames` over `numpy.ndarrays` because of the simple fact that the former allows us to name each feature. This in turn allows us to offer a uniform interface for both single instance and mini-batch learning.

As an example, we will build a simple pipeline that scales the data and trains a logistic regression. Indeed, the `compose.Pipeline` class can be applied to mini-batches, as long as each step is able to do so.


```python
from river import compose
from river import linear_model
from river import preprocessing

model = compose.Pipeline(
    preprocessing.StandardScaler(),
    linear_model.LogisticRegression()
)
```

For this example, we will use `datasets.Higgs`.


```python
from river import datasets

dataset = datasets.Higgs()
if not dataset.is_downloaded:
    dataset.download()
dataset
```




    Higgs dataset.
    
    The data has been produced using Monte Carlo simulations. The first 21 features (columns 2-22)
    are kinematic properties measured by the particle detectors in the accelerator. The last seven
    features are functions of the first 21 features; these are high-level features derived by
    physicists to help discriminate between the two classes.
    
          Name  Higgs                                                                       
          Task  Binary classification                                                       
       Samples  11,000,000                                                                  
      Features  28                                                                          
        Sparse  False                                                                       
          Path  /Users/max.halford/river_data/Higgs/HIGGS.csv.gz                            
           URL  https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
          Size  2.62 GB                                                                     
    Downloaded  True                                                                        



The easiest way to read the data in a mini-batch fashion is to use the `read_csv` from `pandas`.


```python
import pandas as pd

names = [
    'target', 'lepton pT', 'lepton eta', 'lepton phi',
    'missing energy magnitude', 'missing energy phi',
    'jet 1 pt', 'jet 1 eta', 'jet 1 phi', 'jet 1 b-tag',
    'jet 2 pt', 'jet 2 eta', 'jet 2 phi', 'jet 2 b-tag',
    'jet 3 pt', 'jet 3 eta', 'jet 3 phi', 'jet 3 b-tag',
    'jet 4 pt', 'jet 4 eta', 'jet 4 phi', 'jet 4 b-tag',
    'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'
]

for x in pd.read_csv(dataset.path, names=names, chunksize=8096, nrows=3e5):
    y = x.pop('target')
    y_pred = model.predict_proba_many(x)
    model.learn_many(x, y)
```

If you are familiar with scikit-learn, you might be aware that [some](https://scikit-learn.org/dev/computing/scaling_strategies.html#incremental-learning) of their estimators have a `partial_fit` method, which is similar to river's `learn_many` method. Here are some advantages that river has over scikit-learn:

- We guarantee that river's is just as fast, if not faster than scikit-learn. The differences are negligeable, but are slightly in favor of river.
- We take as input dataframes, which allows us to name each feature. The benefit is that you can add/remove/permute features between batches and everything will keep working.
- Estimators that support mini-batches also support single instance learning. This means that you can enjoy the best of both worlds. For instance, you can train with mini-batches and use `predict_one` to make predictions.

Note that you can check which estimators can process mini-batches programmatically:


```python
import importlib
import inspect

def can_mini_batch(obj):
    return hasattr(obj, 'learn_many')

for module in importlib.import_module('river').__all__:
    if module in ['datasets', 'synth']:
        continue
    for name, obj in inspect.getmembers(importlib.import_module(f'river.{module}'), can_mini_batch):
        print(name)
```

    OneClassSVM
    MiniBatchClassifier
    MiniBatchRegressor
    SKL2RiverClassifier
    SKL2RiverRegressor
    Pipeline
    BagOfWords
    TFIDF
    LinearRegression
    LogisticRegression
    Perceptron
    OneVsRestClassifier
    BernoulliNB
    ComplementNB
    MultinomialNB
    MLPRegressor
    StandardScaler


Because mini-batch learning isn't treated as a first-class citizen, some of the river's functionalities require some work in order to play nicely with mini-batches. For instance, the objects from the `metrics` module have an `update` method that take as input a single pair `(y_true, y_pred)`. This might change in the future, depending on the demand.

We plan to promote more models to the mini-batch regime. However, we will only be doing so for the methods that benefit the most from it, as well as those that are most popular. Indeed, `river`'s core philosophy will remain to cater to single instance learning.
