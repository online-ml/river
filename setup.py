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
['pandas>=2.1,<3.0', 'scipy>=1.8.1,<2.0.0']

setup_kwargs = {
    'name': 'river',
    'version': '0.19.0',
    'description': 'Online machine learning in Python',
    'long_description': 'None',
    'author': 'Max Halford',
    'author_email': 'maxhalford25@gmail.com',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.9,<3.13',
}
from build import *
build(setup_kwargs)

setup(**setup_kwargs)
