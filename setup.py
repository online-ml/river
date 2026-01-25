# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['river',
 'river.active',
 'river.anomaly',
 'river.bandit',
 'river.bandit.datasets',
 'river.bandit.envs',
 'river.base',
 'river.checks',
 'river.cluster',
 'river.compat',
 'river.compose',
 'river.conf',
 'river.covariance',
 'river.datasets',
 'river.datasets.synth',
 'river.drift',
 'river.drift.binary',
 'river.drift.datasets',
 'river.ensemble',
 'river.evaluate',
 'river.facto',
 'river.feature_extraction',
 'river.feature_selection',
 'river.forest',
 'river.imblearn',
 'river.linear_model',
 'river.metrics',
 'river.metrics.efficient_rollingrocauc',
 'river.metrics.multioutput',
 'river.misc',
 'river.model_selection',
 'river.multiclass',
 'river.multioutput',
 'river.naive_bayes',
 'river.neighbors',
 'river.neighbors.ann',
 'river.neural_net',
 'river.optim',
 'river.preprocessing',
 'river.proba',
 'river.reco',
 'river.rules',
 'river.sketch',
 'river.stats',
 'river.stream',
 'river.time_series',
 'river.tree',
 'river.tree.mondrian',
 'river.tree.nodes',
 'river.tree.split_criterion',
 'river.tree.splitter',
 'river.utils']

package_data = \
{'': ['*'], 'river.metrics.efficient_rollingrocauc': ['cpp/*']}

install_requires = \
['numpy>=1.23.0', 'pandas>=2.2.3,<3.0.0', 'scipy>=1.14.1,<2.0.0']

setup_kwargs = {
    'name': 'river',
    'version': '0.23.0',
    'description': 'Online machine learning in Python',
    'long_description': '<p align="center">\n  <img height="220px" src="https://github.com/online-ml/river/assets/8095957/64ad5fb6-383c-4bfb-af71-3d055a103a1a" alt="river_logo">\n</p>\n\n<p align="center">\n  <!-- Tests -->\n  <a href="https://github.com/online-ml/river/actions/workflows/unit-tests.yml">\n    <img src="https://github.com/online-ml/river/actions/workflows/unit-tests.yml/badge.svg" alt="unit-tests">\n  </a>\n  <!-- Code quality -->\n  <a href="https://github.com/online-ml/river/actions/workflows/code-quality.yml">\n    <img src="https://github.com/online-ml/river/actions/workflows/code-quality.yml/badge.svg" alt="code-quality">\n  </a>\n  <!-- Documentation -->\n  <a href="https://riverml.xyz">\n    <img src="https://img.shields.io/website?label=docs&style=flat-square&url=https%3A%2F%2Friverml.xyz%2F" alt="documentation">\n  </a>\n  <!-- Discord -->\n  <a href="https://discord.gg/qNmrKEZMAn">\n    <img src="https://dcbadge.vercel.app/api/server/qNmrKEZMAn?style=flat-square" alt="discord">\n  </a>\n  <!-- PyPI -->\n  <a href="https://pypi.org/project/river">\n    <img src="https://img.shields.io/pypi/v/river.svg?label=release&color=blue&style=flat-square" alt="pypi">\n  </a>\n  <!-- PePy -->\n  <a href="https://pepy.tech/project/river">\n    <img src="https://static.pepy.tech/badge/river?style=flat-square" alt="pepy">\n  </a>\n  <!-- Black -->\n  <a href="https://github.com/psf/black">\n    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="black">\n  </a>\n  <!-- Mypy -->\n  <a href="http://mypy-lang.org/">\n    <img src="http://www.mypy-lang.org/static/mypy_badge.svg" alt="mypy">\n  </a>\n  <!-- License -->\n  <a href="https://opensource.org/licenses/BSD-3-Clause">\n    <img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg?style=flat-square" alt="bsd_3_license">\n  </a>\n</p>\n\n</br>\n\n<p align="center">\n  River is a Python library for <a href="https://www.wikiwand.com/en/Online_machine_learning">online machine learning</a>. It aims to be the most user-friendly library for doing machine learning on streaming data. River is the result of a merger between <a href="https://github.com/MaxHalford/creme">creme</a> and <a href="https://github.com/scikit-multiflow/scikit-multiflow">scikit-multiflow</a>.\n</p>\n\n## ⚡️ Quickstart\n\nAs a quick example, we\'ll train a logistic regression to classify the [website phishing dataset](http://archive.ics.uci.edu/ml/datasets/Website+Phishing). Here\'s a look at the first observation in the dataset.\n\n```python\n>>> from pprint import pprint\n>>> from river import datasets\n\n>>> dataset = datasets.Phishing()\n\n>>> for x, y in dataset:\n...     pprint(x)\n...     print(y)\n...     break\n{\'age_of_domain\': 1,\n \'anchor_from_other_domain\': 0.0,\n \'empty_server_form_handler\': 0.0,\n \'https\': 0.0,\n \'ip_in_url\': 1,\n \'is_popular\': 0.5,\n \'long_url\': 1.0,\n \'popup_window\': 0.0,\n \'request_from_other_domain\': 0.0}\nTrue\n\n```\n\nNow let\'s run the model on the dataset in a streaming fashion. We sequentially interleave predictions and model updates. Meanwhile, we update a performance metric to see how well the model is doing.\n\n```python\n>>> from river import compose\n>>> from river import linear_model\n>>> from river import metrics\n>>> from river import preprocessing\n\n>>> model = compose.Pipeline(\n...     preprocessing.StandardScaler(),\n...     linear_model.LogisticRegression()\n... )\n\n>>> metric = metrics.Accuracy()\n\n>>> for x, y in dataset:\n...     y_pred = model.predict_one(x)      # make a prediction\n...     metric.update(y, y_pred)  # update the metric\n...     model.learn_one(x, y)              # make the model learn\n\n>>> metric\nAccuracy: 89.28%\n\n```\n\nOf course, this is just a contrived example. We welcome you to check the [introduction](https://riverml.xyz/dev/introduction/installation/) section of the documentation for a more thorough tutorial.\n\n## 🛠 Installation\n\nRiver is intended to work with **Python 3.10 and above**. Installation can be done with `pip`:\n\n```sh\npip install river\n```\n\nThere are [wheels available](https://pypi.org/project/river/#files) for Linux, MacOS, and Windows. This means you most probably won\'t have to build River from source.\n\nYou can install the latest development version from GitHub as so:\n\n```sh\npip install git+https://github.com/online-ml/river --upgrade\npip install git+ssh://git@github.com/online-ml/river.git --upgrade  # using SSH\n```\n\nThis method requires having Cython and Rust installed on your machine.\n\n## 🔮 Features\n\nRiver provides online implementations of the following family of algorithms:\n\n- Linear models, with a wide array of optimizers\n- Decision trees and random forests\n- (Approximate) nearest neighbors\n- Anomaly detection\n- Drift detection\n- Recommender systems\n- Time series forecasting\n- Bandits\n- Factorization machines\n- Imbalanced learning\n- Clustering\n- Bagging/boosting/stacking\n- Active learning\n\nRiver also provides other online utilities:\n\n- Feature extraction and selection\n- Online statistics and metrics\n- Preprocessing\n- Built-in datasets\n- Progressive model validation\n- Model pipelines\n\nCheck out [the API](https://riverml.xyz/latest/api/overview/) for a comprehensive overview\n\n## 🤔 Should I be using River?\n\nYou should ask yourself if you need online machine learning. The answer is likely no. Most of the time batch learning does the job just fine. An online approach might fit the bill if:\n\n- You want a model that can learn from new data without having to revisit past data.\n- You want a model which is robust to [concept drift](https://www.wikiwand.com/en/Concept_drift).\n- You want to develop your model in a way that is closer to what occurs in a production context, which is usually event-based.\n\nSome specificities of River are that:\n\n- It focuses on clarity and user experience, more so than performance.\n- It\'s very fast at processing one sample at a time. Try it, you\'ll see.\n- It plays nicely with the rest of Python\'s ecosystem.\n\n## 🔗 Useful links\n\n- [Documentation](https://riverml.xyz)\n- [Package releases](https://pypi.org/project/river/#history)\n- [awesome-online-machine-learning](https://github.com/online-ml/awesome-online-machine-learning)\n- [2022 presentation at GAIA](https://www.youtube.com/watch?v=nzFTmJnIakk&list=PLIU25-FciwNaz5PqWPiHmPCMOFYoEsJ8c&index=5)\n- [Online Clustering: Algorithms, Evaluation, Metrics, Applications and Benchmarking](https://dl.acm.org/doi/10.1145/3534678.3542600) from [KDD\'22](https://kdd.org/kdd2022/).\n\n## 👐 Contributing\n\nFeel free to contribute in any way you like, we\'re always open to new ideas and approaches.\n\n- [Open a discussion](https://github.com/online-ml/river/discussions/new) if you have any question or enquiry whatsoever. It\'s more useful to ask your question in public rather than sending us a private email. It\'s also encouraged to open a discussion before contributing, so that everyone is aligned and unnecessary work is avoided.\n- Feel welcome to [open an issue](https://github.com/online-ml/river/issues/new/choose) if you think you\'ve spotted a bug or a performance issue.\n- Our [roadmap](https://github.com/orgs/online-ml/projects/3?query=is%3Aopen+sort%3Aupdated-desc) is public. Feel free to work on anything that catches your eye, or to make suggestions.\n\nPlease check out the [contribution guidelines](https://github.com/online-ml/river/blob/main/CONTRIBUTING.md) if you want to bring modifications to the code base.\n\n## 🤝 Affiliations\n\n<p align="center">\n  <img width="70%" src="https://docs.google.com/drawings/d/e/2PACX-1vSagEhWAjDsb0c24En_fhWAf9DJZbyh5YjU7lK0sNowD2m9uv9TuFm-U77k6ObqTyN2mP05Avf6TCJc/pub?w=2073&h=1127" alt="affiliations">\n</p>\n\n## 💬 Citation\n\nIf River has been useful to you, and you would like to cite it in a scientific publication, please refer to the [paper](https://www.jmlr.org/papers/volume22/20-1380/20-1380.pdf) published at JMLR:\n\n```bibtex\n@article{montiel2021river,\n  title={River: machine learning for streaming data in Python},\n  author={Montiel, Jacob and Halford, Max and Mastelini, Saulo Martiello\n          and Bolmier, Geoffrey and Sourty, Raphael and Vaysse, Robin and Zouitine, Adil\n          and Gomes, Heitor Murilo and Read, Jesse and Abdessalem, Talel and others},\n  year={2021}\n}\n```\n\n## 📝 License\n\nRiver is free and open-source software licensed under the [3-clause BSD license](https://github.com/online-ml/river/blob/main/LICENSE).\n',
    'author': 'Max Halford',
    'author_email': 'maxhalford25@gmail.com',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'https://riverml.xyz/',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.10,<4.0',
}
from build import *
build(setup_kwargs)

setup(**setup_kwargs)
