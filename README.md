<p align="center">
  <img height="200px" src="https://docs.google.com/drawings/d/e/2PACX-1vSl80T4MnWRsPX3KvlB2kn6zVdHdUleG_w2zBiLS7RxLGAHxiSYTnw3LZtXh__YMv6KcIOYOvkSt9PB/pub?w=447&h=182" alt="creme_logo">
</p>

<p align="center">
  <!-- Travis -->
  <a href="https://travis-ci.org/creme-ml/creme">
    <img src="https://img.shields.io/travis/creme-ml/creme/master.svg?style=flat-square" alt="travis">
  </a>
  <!-- Codecov -->
  <a href="https://codecov.io/gh/creme-ml/creme">
    <img src="https://img.shields.io/codecov/c/gh/creme-ml/creme.svg?style=flat-square" alt="codecov">
  </a>
  <!-- Documentation -->
  <a href="https://creme-ml.github.io/">
    <img src="https://img.shields.io/website?label=documentation&style=flat-square&url=https%3A%2F%2Fcreme-ml.github.io%2F" alt="documentation">
  </a>
  <!-- Gitter -->
  <a href="https://gitter.im/creme-ml/community?utm_source=share-link&utm_medium=link&utm_campaign=share-link">
    <img src="https://img.shields.io/gitter/room/creme-ml/community?color=blueviolet&style=flat-square" alt="gitter">
  </a>
  <!-- PyPI -->
  <a href="https://pypi.org/project/creme">
    <img src="https://img.shields.io/pypi/v/creme.svg?label=release&color=blue&style=flat-square" alt="pypi">
  </a>
  <!-- PePy -->
  <a href="https://pepy.tech/project/creme">
    <img src="https://img.shields.io/badge/dynamic/json?style=flat-square&maxAge=86400&label=downloads&query=%24.total_downloads&url=https%3A%2F%2Fapi.pepy.tech%2Fapi%2Fprojects%2Fcreme" alt="pepy">
  </a>
  <!-- License -->
  <a href="https://opensource.org/licenses/BSD-3-Clause">
    <img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg?style=flat-square" alt="bsd_3_license">
  </a>
</p>

<p align="center">
  <code>creme</code> is a Python library for <a href="https://www.wikiwand.com/en/Online_machine_learning">online machine learning</a>. All the tools in the library can be updated with a single observation at a time, and can therefore be used to <b>learn from streaming data</b>.
</p>

## 🤗 Merger announcement

### TLDR

[creme](https://creme-ml.github.io/) and [scikit-multiflow](https://scikit-multiflow.github.io/) are merging. A new package will result from this merge. Both development teams will work together on this new package.

### Why?

We feel that both projects share the same vision. We believe that pooling our resources instead of duplicating work will benefit both sides. We are also confident that this will benefit both communities. There will be more people working on the new project, which will allow us to distribute work more efficiently. We will thus be able to work on more features and improve the overall quality of the project.

### How does this affect each project?

Both projects will stop active development once the new package is released. The code for both projects will remain publicly available, although development will only focus on minor maintenance during a transition period. During this transition period, most of the functionality of both projects will be made available in the new package.

The architecture of the new package is more Pythonic. It will focus on single-instance incremental models. The new API reflects these changes.

Detailed information on the new architecture and API will be available with the release of the new package.

### How does this affect users?

We encourage users to move towards the new package when possible. We understand that this transition will require an extra effort in the short term from current users. However, we believe that the result will be better for everyone in the long run.

You will still be able to install and use `creme` as well as `scikit-multiflow`. Both projects will remain on PyPI, conda-forge and GitHub.

### When?

The target date for the first release: **2nd half of October 2020**.

## ⚡️Quickstart

As a quick example, we'll train a logistic regression to classify the [website phishing dataset](http://archive.ics.uci.edu/ml/datasets/Website+Phishing). Here's a look at the first observation in the dataset.

```python
>>> from pprint import pprint
>>> from creme import datasets

>>> X_y = datasets.Phishing()  # this is a generator

>>> for x, y in X_y:
...     pprint(x)
...     print(y)
...     break
{'age_of_domain': 1,
 'anchor_from_other_domain': 0.0,
 'empty_server_form_handler': 0.0,
 'https': 0.0,
 'ip_in_url': 1,
 'is_popular': 0.5,
 'long_url': 1.0,
 'popup_window': 0.0,
 'request_from_other_domain': 0.0}
True

```

Now let's run the model on the dataset in a streaming fashion. We sequentially interleave predictions and model updates. Meanwhile, we update a performance metric to see how well the model is doing.

```python
>>> from creme import compose
>>> from creme import linear_model
>>> from creme import metrics
>>> from creme import preprocessing

>>> model = compose.Pipeline(
...     preprocessing.StandardScaler(),
...     linear_model.LogisticRegression()
... )

>>> metric = metrics.Accuracy()

>>> for x, y in X_y:
...     y_pred = model.predict_one(x)      # make a prediction
...     metric = metric.update(y, y_pred)  # update the metric
...     model = model.fit_one(x, y)        # make the model learn

>>> metric
Accuracy: 89.20%

```

## 🛠 Installation

`creme` is intended to work with **Python 3.6 or above**. Installation can be done with `pip`:

```sh
pip install creme
```

There are [wheels available](https://pypi.org/project/creme/#files) for Linux, MacOS, and Windows, which means that in most cases you won't have to build `creme` from source.

You can install the latest development version from GitHub as so:

```sh
pip install git+https://github.com/creme-ml/creme --upgrade
```

Or, through SSH:

```sh
pip install git+ssh://git@github.com/creme-ml/creme.git --upgrade
```

## 🧠 Philosophy

Machine learning is often done in a batch setting, whereby a model is fitted to a dataset in one go. This results in a static model which has to be retrained in order to learn from new data. In many cases, this isn't elegant nor efficient, and usually incurs [a fair amount of technical debt](https://research.google/pubs/pub43146/). Indeed, if you're using a batch model, then you need to think about maintaining a training set, monitoring real-time performance, model retraining, etc.

With `creme`, we encourage a different approach, which is to continuously learn a stream of data. This means that the model process one observation at a time, and can therefore be updated on the fly. This allows to learn from massive datasets that don't fit in main memory. Online machine learning also integrates nicely in cases where new data is constantly arriving. It shines in many use cases, such as time series forecasting, spam filtering, recommender systems, CTR prediction, and IoT applications. If you're bored with retraining models and want to instead build dynamic models, then online machine learning (and therefore `creme`!) might be what you're looking for.

Here are some benefits of using `creme` (and online machine learning in general):

- **Incremental**: models can update themselves in real-time.
- **Adaptive**: models can adapt to [concept drift](https://www.wikiwand.com/en/Concept_drift).
- **Production-ready**: working with data streams makes it simple to replicate production scenarios during model development.
- **Efficient**: models don't have to be retrained and require little compute power, which [lowers their carbon footprint](https://arxiv.org/abs/1907.10597)
- **Fast**: when the goal is to learn and predict with a single instance at a time, then `creme` is a order of magnitude faster than PyTorch, Tensorflow, and scikit-learn.

## 🔥 Features

- Linear models with a wide array of optimizers
- Nearest neighbors, decision trees, naïve Bayes
- [Progressive model validation](https://hunch.net/~jl/projects/prediction_bounds/progressive_validation/coltfinal.pdf)
- Model pipelines as a first-class citizen
- Anomaly detection
- Recommender systems
- Time series forecasting
- Imbalanced learning
- Clustering
- Feature extraction and selection
- Online statistics and metrics
- Built-in datasets
- And [much more](https://creme-ml.github.io/content/api.html)

## 🔗 Useful links

- [Documentation](https://creme-ml.github.io/)
- [Benchmarks](https://github.com/creme-ml/creme/tree/master/benchmarks)
- [Issue tracker](https://github.com/creme-ml/creme/issues)
- [Package releases](https://pypi.org/project/creme/#history)

## 👁️ Media

- PyData Amsterdam 2019 presentation ([slides](https://maxhalford.github.io/slides/creme-pydata/), [video](https://www.youtube.com/watch?v=P3M6dt7bY9U&list=PLGVZCDnMOq0q7_6SdrC2wRtdkojGBTAht&index=11))
- [Toulouse Data Science Meetup presentation](https://maxhalford.github.io/slides/creme-tds/)
- [Machine learning for streaming data with creme](https://towardsdatascience.com/machine-learning-for-streaming-data-with-creme-dacf5fb469df)
- [Hong Kong Data Science Meetup presentation](https://maxhalford.github.io/slides/hkml2020.pdf)

## 👍 Contributing

Feel free to contribute in any way you like, we're always open to new ideas and approaches. You can also take a look at the [issue tracker](https://github.com/creme-ml/creme/issues) and the [icebox](https://github.com/creme-ml/creme/projects/2) to see if anything takes your fancy. Please check out the [contribution guidelines](https://github.com/creme-ml/creme/blob/master/CONTRIBUTING.md) if you want to bring modifications to the code base. You can view the list of people who have contributed [here](https://github.com/creme-ml/creme/graphs/contributors).

## 💬 Citation

Please use the following citation if you want to reference creme in a scientific publication:

```
@software{creme,
  title = {{creme}, a {P}ython library for online machine learning},
  author = {Halford, Max and Bolmier, Geoffrey and Sourty, Raphael and Vaysse, Robin and Zouitine, Adil},
  url = {https://github.com/creme-ml/creme},
  version = {0.6.1},
  date = {2020-06-10},
  year = {2019}
}
```

Note that in the future we will probably publish a dedicated research paper.

## 📝 License

`creme` is free and open-source software licensed under the [3-clause BSD license](https://github.com/creme-ml/creme/blob/master/LICENSE).
