<p align="center">
  <img height="220px" src="docs/img/logo.svg" alt="river_logo">
</p>

<p align="center">
  <!-- Tests -->
  <a href="https://github.com/online-ml/river/actions/workflows/ci.yml">
    <img src="https://github.com/online-ml/river/actions/workflows/ci.yml/badge.svg" alt="CI Pipeline">
  </a>
  <!-- Documentation -->
  <a href="https://riverml.xyz">
    <img src="https://img.shields.io/website?label=docs&style=flat-square&url=https%3A%2F%2Friverml.xyz%2F" alt="documentation">
  </a>
  <!-- Discord -->
  <a href="https://discord.gg/qNmrKEZMAn">
    <img src="https://dcbadge.vercel.app/api/server/qNmrKEZMAn?style=flat-square" alt="discord">
  </a>
  <!-- Roadmap -->
  <a href="https://github.com/orgs/online-ml/projects/3/">
    <img src="https://img.shields.io/website?label=roadmap&style=flat-square&url=https://github.com/orgs/online-ml/projects/3/" alt="roadmap">
  </a>
  <!-- PyPI -->
  <a href="https://pypi.org/project/river">
    <img src="https://img.shields.io/pypi/v/river.svg?label=release&color=blue&style=flat-square" alt="pypi">
  </a>
  <!-- PePy -->
  <a href="https://pepy.tech/project/river">
    <img src="https://static.pepy.tech/badge/river?style=flat-square" alt="pepy">
  </a>
  <!-- Mypy -->
  <a href="http://mypy-lang.org/">
    <img src="http://www.mypy-lang.org/static/mypy_badge.svg" alt="mypy">
  </a>
  <!-- License -->
  <a href="https://opensource.org/licenses/BSD-3-Clause">
    <img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg?style=flat-square" alt="bsd_3_license">
  </a>
</p>

</br>

<p align="center">
  River is a Python library for <a href="https://www.wikiwand.com/en/Online_machine_learning">online machine learning</a>. It aims to be the most user-friendly library for doing machine learning on streaming data. River is the result of a merger between <a href="https://github.com/MaxHalford/creme">creme</a> and <a href="https://github.com/scikit-multiflow/scikit-multiflow">scikit-multiflow</a>.
</p>

## ⚡️ Quickstart

As a quick example, we'll train a logistic regression to classify the [website phishing dataset](http://archive.ics.uci.edu/ml/datasets/Website+Phishing). Here's a look at the first observation in the dataset.

```python
>>> from pprint import pprint
>>> from river import datasets

>>> dataset = datasets.Phishing()

>>> for x, y in dataset:
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
>>> from river import compose
>>> from river import linear_model
>>> from river import metrics
>>> from river import preprocessing

>>> model = compose.Pipeline(
...     preprocessing.StandardScaler(),
...     linear_model.LogisticRegression()
... )

>>> metric = metrics.Accuracy()

>>> for x, y in dataset:
...     y_pred = model.predict_one(x)      # make a prediction
...     metric = metric.update(y, y_pred)  # update the metric
...     model = model.learn_one(x, y)      # make the model learn

>>> metric
Accuracy: 89.20%

```

Of course, this is just a contrived example. We welcome you to check the [introduction](https://riverml.xyz/dev/introduction/installation/) section of the documentation for a more thorough tutorial.

## 🛠 Installation

River is intended to work with **Python 3.8 and above**. Installation can be done with `pip`:

```sh
pip install river
```

There are [wheels available](https://pypi.org/project/river/#files) for Linux, MacOS, and Windows, which means that you most probably won't have to build River from source.

You can install the latest development version from GitHub as so:

```sh
pip install git+https://github.com/online-ml/river --upgrade
```

Or, through SSH:

```sh
pip install git+ssh://git@github.com/online-ml/river.git --upgrade
```

## 🔮 Features

- Linear models with a wide array of optimizers
- Nearest neighbors, decision trees, naïve Bayes
- Anomaly detection
- Drift detection
- Recommender systems
- Time series forecasting
- Imbalanced learning
- Clustering
- Feature extraction and selection
- Online statistics and metrics
- Built-in datasets
- Progressive model validation
- Model pipelines as a first-class citizen
- Check out [the API](https://riverml.xyz/latest/api/overview/) for a comprehensive overview

## 🤔 Should I be using River?

You should ask yourself if you need online machine learning. The answer is likely no. Most of the time batch learning does the job just fine. An online approach might fit the bill if:

- You want a model that can learn from new data without having to revisit past data.
- You want a model which is robust to [concept drift](https://www.wikiwand.com/en/Concept_drift).
- You want to develop your model in a way that is closer to what occurs in a production context, which is usually event-based.

Some specificities of River are that:

- It focuses on clarity and user experience, more so than performance.
- It's very fast at processing one sample at a time. Try it, you'll see.
- It plays nicely with the rest of Python's ecosystem.

## 🔗 Useful links

- [Documentation](https://riverml.xyz)
- [Package releases](https://pypi.org/project/river/#history)
- [awesome-online-machine-learning](https://github.com/online-ml/awesome-online-machine-learning)
- [2022 presentation at GAIA](https://www.youtube.com/watch?v=nzFTmJnIakk&list=PLIU25-FciwNaz5PqWPiHmPCMOFYoEsJ8c&index=5)
- [Online Clustering: Algorithms, Evaluation, Metrics, Applications and Benchmarking](https://dl.acm.org/doi/10.1145/3534678.3542600) from [KDD'22](https://kdd.org/kdd2022/).

## 👐 Contributing

Feel free to contribute in any way you like, we're always open to new ideas and approaches.

- [Open a discussion](https://github.com/online-ml/river/discussions/new) if you have any question or enquiry whatsoever. It's more useful to ask your question in public rather than sending us a private email. It's also encouraged to open a discussion before contributing, so that everyone is aligned and unnecessary work is avoided.
- Feel welcome to [open an issue](https://github.com/online-ml/river/issues/new/choose) if you think you've spotted a bug or a performance issue.
- Our [roadmap](https://github.com/orgs/online-ml/projects/3?query=is%3Aopen+sort%3Aupdated-desc) is public. Feel free to work on anything that catches your eye, or to make suggestions.

Please check out the [contribution guidelines](https://github.com/online-ml/river/blob/main/CONTRIBUTING.md) if you want to bring modifications to the code base.

## 🤝 Affiliations

<p align="center">
  <img width="70%" src="https://docs.google.com/drawings/d/e/2PACX-1vSagEhWAjDsb0c24En_fhWAf9DJZbyh5YjU7lK0sNowD2m9uv9TuFm-U77k6ObqTyN2mP05Avf6TCJc/pub?w=2073&h=1127" alt="affiliations">
</p>

## 💬 Citation

If River has been useful to you and you would like to cite it in an scientific publication, please refer to the [paper](https://www.jmlr.org/papers/volume22/20-1380/20-1380.pdf) published at JMLR:

```bibtex
@article{montiel2021river,
  title={River: machine learning for streaming data in Python},
  author={Montiel, Jacob and Halford, Max and Mastelini, Saulo Martiello
          and Bolmier, Geoffrey and Sourty, Raphael and Vaysse, Robin and Zouitine, Adil
          and Gomes, Heitor Murilo and Read, Jesse and Abdessalem, Talel and others},
  year={2021}
}
```

## 📝 License

River is free and open-source software licensed under the [3-clause BSD license](https://github.com/online-ml/river/blob/main/LICENSE).
