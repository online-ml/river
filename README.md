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

## ⚡️Quickstart

As a quick example, we'll train a logistic regression to classify the [website phishing dataset](http://archive.ics.uci.edu/ml/datasets/Website+Phishing). Here's a look at the first observation in the dataset.

```python
>>> from pprint import pprint
>>> from creme import datasets

>>> X_y = datasets.Phishing()

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

Now let's run the model on the dataset in a streaming fashion. We will sequentially make predictions and model updates. Meanwhile we will update a performance metric to see how well the model is doing.

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
Accuracy: 89.28%

```

## 🛠 Installation

`creme` is intended to work with Python 3.6 or above. Installation can be done by using `pip`:

    pip install creme

There are [wheels available](https://pypi.org/project/creme/#files) for Linux, MacOS, and Windows. You can also install the latest development version as so:

    pip install git+https://github.com/creme-ml/creme

    # Or, through SSH:
    pip install git+ssh://git@github.com/creme-ml/creme.git

Note that installing the development version requires already having [Cython](https://github.com/cython/cython) installed.

## 🧠 Philosophy

Machine learning is often done in a batch setting, whereby a model is fitted to a dataset in one go. This results in a static model which has to be retrained in order to learn from new data. In many cases, this isn't elegant nor efficient, and usually incurs [a fair amount of technical debt](https://research.google/pubs/pub43146/). Indeed, if you're using a batch model, then you need to think about maintaining a training set, monitoring real-time performance, model retraining, etc.

With `creme`, we encourage a different approach, which is to fit a model to a stream of data. This means that the model learns from one observation at a time, and can therefore be updated on the fly. This allows to learn from massive datasets that don't fit in main memory. Online machine learning also integrates nicely in cases where new data is constantly arriving. It shines in many use cases, such as time series forecasting, spam filtering, recommender systems, CTR prediction, and IoT applications. If you're bored with retraining models and want to instead build dynamic models, then online machine learning (and therefore `creme`!) might be what you're looking for.

Here are some benefits of using `creme` (and online machine learning in general):

- **Incremental**: models can update themselves in real-time.
- **Adaptive**: models can adapt to [concept drift](https://www.wikiwand.com/en/Concept_drift).
- **Production-ready**: working with data streams makes it simple to replicate production scenarios during model development.
- **Efficient**: models don't have to be retrained and require little compute power, which [lowers their carbon footprint](https://arxiv.org/abs/1907.10597)

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
  - [API reference](https://creme-ml.github.io/content/api.html)
  - [User guide](https://creme-ml.github.io/content/user-guide.html)
  - [FAQ](https://creme-ml.github.io/content/faq.html)
  - [Change history](https://creme-ml.github.io/content/changelog.html)
- [Benchmarks](https://github.com/creme-ml/creme/tree/master/benchmarks)
- [Issue tracker](https://github.com/creme-ml/creme/issues)
- [Package releases](https://pypi.org/project/creme/#history)

## 💬 Media

- PyData Amsterdam 2019 presentation ([slides](https://maxhalford.github.io/slides/creme-pydata/), [video](https://www.youtube.com/watch?v=P3M6dt7bY9U&list=PLGVZCDnMOq0q7_6SdrC2wRtdkojGBTAht&index=11))
- [Toulouse Data Science presentation](https://maxhalford.github.io/slides/creme-tds/)
- [Blog post on pyimagesearch](https://www.pyimagesearch.com/2019/06/17/online-incremental-learning-with-keras-and-creme/)

## 👍 Contributing

Feel free to contribute in any way you like, we're always open to new ideas and approaches. If you want to contribute to the code base please check out the [CONTRIBUTING.md file](https://github.com/creme-ml/creme/blob/master/CONTRIBUTING.md). Also take a look at the [issue tracker](https://github.com/creme-ml/creme/issues) and see if anything takes your fancy.

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Again, contributions of any kind are welcome!

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://maxhalford.github.io"><img src="https://avatars1.githubusercontent.com/u/8095957?v=4" width="100px;" alt="Max Halford"/><br /><sub><b>Max Halford</b></sub></a><br /><a href="#projectManagement-MaxHalford" title="Project Management">📆</a> <a href="https://github.com/creme-ml/creme/commits?author=MaxHalford" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/AdilZouitine"><img src="https://avatars0.githubusercontent.com/u/24889239?v=4" width="100px;" alt="AdilZouitine"/><br /><sub><b>AdilZouitine</b></sub></a><br /><a href="https://github.com/creme-ml/creme/commits?author=AdilZouitine" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/raphaelsty"><img src="https://avatars3.githubusercontent.com/u/24591024?v=4" width="100px;" alt="Raphael Sourty"/><br /><sub><b>Raphael Sourty</b></sub></a><br /><a href="https://github.com/creme-ml/creme/commits?author=raphaelsty" title="Code">💻</a></td>
    <td align="center"><a href="http://www.linkedin.com/in/gbolmier"><img src="https://avatars0.githubusercontent.com/u/25319692?v=4" width="100px;" alt="Geoffrey Bolmier"/><br /><sub><b>Geoffrey Bolmier</b></sub></a><br /><a href="https://github.com/creme-ml/creme/commits?author=gbolmier" title="Code">💻</a></td>
    <td align="center"><a href="http://koaning.io"><img src="https://avatars1.githubusercontent.com/u/1019791?v=4" width="100px;" alt="vincent d warmerdam "/><br /><sub><b>vincent d warmerdam </b></sub></a><br /><a href="https://github.com/creme-ml/creme/commits?author=koaning" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/VaysseRobin"><img src="https://avatars2.githubusercontent.com/u/32324822?v=4" width="100px;" alt="VaysseRobin"/><br /><sub><b>VaysseRobin</b></sub></a><br /><a href="https://github.com/creme-ml/creme/commits?author=VaysseRobin" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/tweakyllama"><img src="https://avatars0.githubusercontent.com/u/7049400?v=4" width="100px;" alt="Lygon Bowen-West"/><br /><sub><b>Lygon Bowen-West</b></sub></a><br /><a href="https://github.com/creme-ml/creme/commits?author=tweakyllama" title="Code">💻</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/flegac"><img src="https://avatars2.githubusercontent.com/u/4342302?v=4" width="100px;" alt="Florent Le Gac"/><br /><sub><b>Florent Le Gac</b></sub></a><br /><a href="https://github.com/creme-ml/creme/commits?author=flegac" title="Code">💻</a></td>
    <td align="center"><a href="http://www.pyimagesearch.com"><img src="https://avatars2.githubusercontent.com/u/759645?v=4" width="100px;" alt="Adrian Rosebrock"/><br /><sub><b>Adrian Rosebrock</b></sub></a><br /><a href="#blog-jrosebr1" title="Blogposts">📝</a></td>
    <td align="center"><a href="https://github.com/JovanVeljanoski"><img src="https://avatars1.githubusercontent.com/u/18574951?v=4" width="100px;" alt="Jovan Veljanoski"/><br /><sub><b>Jovan Veljanoski</b></sub></a><br /><a href="https://github.com/creme-ml/creme/commits?author=JovanVeljanoski" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/d-me-tree"><img src="https://avatars1.githubusercontent.com/u/4955958?v=4" width="100px;" alt="Dimitri"/><br /><sub><b>Dimitri</b></sub></a><br /><a href="https://github.com/creme-ml/creme/commits?author=d-me-tree" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/greatsharma"><img src="https://avatars0.githubusercontent.com/u/32649388?v=4" width="100px;" alt="Gaurav Sharma"/><br /><sub><b>Gaurav Sharma</b></sub></a><br /><a href="https://github.com/creme-ml/creme/commits?author=greatsharma" title="Code">💻</a></td>
  </tr>
</table>
<!-- markdownlint-enable -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

## 📝 License

`creme` is free and open-source software licensed under the [3-clause BSD license](https://github.com/creme-ml/creme/blob/master/LICENSE).
