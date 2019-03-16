<div align="center">
  <img height="240px" src="https://docs.google.com/drawings/d/e/2PACX-1vSl80T4MnWRsPX3KvlB2kn6zVdHdUleG_w2zBiLS7RxLGAHxiSYTnw3LZtXh__YMv6KcIOYOvkSt9PB/pub?w=841&h=350" alt="creme_logo"/>
</div>

<div align="center">
  <!-- Travis -->
  <a href="https://travis-ci.org/creme-ml/creme">
    <img src="https://img.shields.io/travis/creme-ml/creme/master.svg?style=for-the-badge" alt="travis" />
  </a>
  <!-- Codecov -->
  <a href="https://codecov.io/gh/creme-ml/creme">
    <img src="https://img.shields.io/codecov/c/gh/creme-ml/creme.svg?style=for-the-badge" alt="codecov" />
  </a>
  <!-- PyPI -->
  <a href="https://pypi.org/project/creme">
    <img src="https://img.shields.io/pypi/v/creme.svg?style=for-the-badge" alt="pypi" />
  </a>
  <!-- License -->
  <a href="https://opensource.org/licenses/BSD-3-Clause">
    <img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg?style=for-the-badge" alt="bsd_3_license"/>
  </a>
</div>

<br/>

`creme` is a library for in**creme**ntal learning. Incremental learning is a machine learning regime where the observations are made available one by one. It is also known as online learning, iterative learning, or sequential learning. This is in contrast to batch learning where all the data is processed at once. Incremental learning is desirable when the data is too big to fit in memory, or simply when it isn't available all at once. `creme`'s API is heavily inspired from that of [scikit-learn](https://scikit-learn.org/stable/), enough so that users who are familiar with it should feel right at home.

## Useful links

- [Documentation](https://creme-ml.github.io/) (includes tutorial notebooks)
- [Issue tracker](https://github.com/creme-ml/creme/issues)
- [Package releases](https://pypi.org/project/creme/#history)
- [Change history](CHANGELOG.md)

## Installation

:warning: `creme` requires Python 3.6 or above.

`creme` mostly relies on Python's standard library. However, it sometimes relies on `numpy`, `scipy`, and `scikit-learn` in order to avoid reinventing the wheel. `creme` can simply be installed with `pip`.

```sh
pip install creme
```

## Quick example

In the following snippet we'll be fitting an online logistic regression. The weights of the model will be optimized with the [AdaGrad](http://akyrillidis.github.io/notes/AdaGrad) algorithm. We'll scale the data so that each variable has a mean of 0 and a standard deviation of 1. The standard scaling and the logistic regression are combined using a compose. We'll be using the `stream.iter_sklearn_dataset` function for streaming over the [Wisconsin breast cancer dataset](http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29). We'll measure the ROC AUC using [progressive validation](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.153.3925&rep=rep1&type=pdf).

```python
>>> from creme import compose
>>> from creme import linear_model
>>> from creme import metrics
>>> from creme import model_selection
>>> from creme import optim
>>> from creme import preprocessing
>>> from creme import stream
>>> from sklearn import datasets

>>> X_y = stream.iter_sklearn_dataset(
...     load_dataset=datasets.load_breast_cancer,
...     shuffle=True,
...     random_state=42
... )
>>> optimizer = optim.AdaGrad()
>>> model = compose.Pipeline([
...     ('scale', preprocessing.StandardScaler()),
...     ('learn', linear_model.LogisticRegression(optimizer))
... ])
>>> metric = metrics.F1Score()

>>> model_selection.online_score(X_y, model, metric)
F1Score: 0.97191

```

## Comparison with other solutions

- [scikit-learn](https://scikit-learn.org/stable/): Some of it's estimators have a `partial_fit` method which allows them to update themselves with new observations. However, online learning isn't a first class citizen, which can make it a bit awkward to put a streaming pipeline in place. You should definitely use scikit-learn if your data fits in memory and that you can afford retraining your model from scratch when you have new data to train on.
- [Vowpal Wabbit](https://github.com/VowpalWabbit/vowpal_wabbit/wiki): VW is probably the fastest out-of-core learning system available. At it's core it implements a state-of-the-art adaptive gradient descent algorithm with many tricks. It also has some mechanisms for doing [active learning](https://www.wikiwand.com/en/Active_learning_(machine_learning)) and using [bandits](https://www.wikiwand.com/en/Multi-armed_bandit). However it isn't a "true" online learning system as it assumes the data is available in a file and can looped over multiple times. Also it is somewhat difficult to [grok](https://www.wikiwand.com/en/Grok) for newcomers.
- [LIBOL](https://github.com/LIBOL/SOL): This is a good library written by academics with some great documentation. It's written in C++ and seems to be pretty fast. However it only focuses on the learning aspect of online learning, not on other mundane yet useful tasks such as feature extraction and preprocessing. Moreover it hasn't been updated for a few years.
- [Spark Streaming](https://spark.apache.org/docs/latest/streaming-programming-guide.html): This is an extension of [Apache Spark](https://www.wikiwand.com/en/Apache_Spark) which caters to big data practitioners. It provides a lot of practical tools for manipulating streaming data in it's true sense. It also has some compatibility with the [MLlib](https://spark.apache.org/docs/latest/ml-guide.html) for implementing online learning algorithms, such as [streaming linear regression](https://spark.apache.org/docs/latest/mllib-linear-methods.html#streaming-linear-regression) and [streaming k-means](https://spark.apache.org/docs/latest/mllib-clustering.html#streaming-k-means). However it is a somewhat overwhelming solution which might be a bit overkill for certain use cases.
- [TensorFlow](https://www.wikiwand.com/en/TensorFlow): Deep learning systems are in some sense online learning systems. Indeed it is possible to put in place a DL pipeline for learning from incoming observations. Because frameworks such as [Keras](https://keras.io/) and [PyTorch](https://pytorch.org/) are popular and well-backed, there is no real point in implementing neural networks in creme. For a lot of problems neural networks might not be the right tool, and you might want to use a simple logistic regression or a decision tree (for which online algorithms exist).

Feel free to open an issue if you feel like other solutions are worth mentioning.

## Contributing

Like many subfields of machine learning, online learning is far from being an exact science and so there is still a lot to do. Feel free to contribute in any way you like, we're always open to new ideas and approaches. Here are some specific points where help would be particularly appreciated:

- Finding bottlenecks and using [Cython](https://cython.org/) when applicable
- Identifying parts of the documentation that are hard to understand or are not very well written
- Anything related to the [issue tracker](https://github.com/creme-ml/creme/issues)

Last but not least you are more than welcome to share with us how you're using `creme` or online learning in general! We believe that online learning solves a lot of pain points in practice and would love to share experiences.

## License

See the [license file](LICENSE).
