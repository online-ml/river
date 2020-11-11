<img src="https://raw.githubusercontent.com/scikit-multiflow/scikit-multiflow/master/docs/_static/images/skmultiflow-logo-wide.png" height="100"/>

[![Build status](https://travis-ci.org/scikit-multiflow/scikit-multiflow.svg?branch=master)](https://travis-ci.org/scikit-multiflow/scikit-multiflow)
[![Build Status](https://dev.azure.com/scikit-multiflow/scikit-multiflow/_apis/build/status/scikit-multiflow.scikit-multiflow?branchName=master)](https://dev.azure.com/scikit-multiflow/scikit-multiflow/_build/latest?definitionId=1&branchName=master)
[![codecov](https://codecov.io/gh/scikit-multiflow/scikit-multiflow/branch/master/graph/badge.svg)](https://codecov.io/gh/scikit-multiflow/scikit-multiflow)
![Python version](https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7%20%7C%203.8-blue.svg)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/scikit-multiflow/badges/platforms.svg)](https://anaconda.org/conda-forge/scikit-multiflow)
[![PyPI version](https://badge.fury.io/py/scikit-multiflow.svg)](https://badge.fury.io/py/scikit-multiflow)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/scikit-multiflow/badges/version.svg)](https://anaconda.org/conda-forge/scikit-multiflow)
[![DockerHub](https://img.shields.io/badge/docker-available-blue.svg?logo=docker)](https://hub.docker.com/r/skmultiflow/scikit-multiflow)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Gitter](https://badges.gitter.im/scikit-multiflow/community.svg)](https://gitter.im/scikit-multiflow/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

`scikit-multiflow` is a machine learning package for streaming data in Python.

### Quick links
* [Webpage](https://scikit-multiflow.github.io/)
* [Documentation](https://scikit-multiflow.readthedocs.io/en/stable/)
* [Community](https://scikit-multiflow.github.io/community/)

# Features

### Incremental Learning
Stream learning models are created incrementally and are updated continuously. They are suitable
for big data applications where real-time response is vital.

### Adaptive learning
Changes in data distribution harm learning. Adaptive methods are specifically designed to be
robust to concept drift changes in dynamic environments.

### Resource-wise efficient
Streaming techniques efficiently handle resources such as memory and processing time given the
unbounded nature of data streams.

### Easy to use
scikit-multiflow is designed for users with any experience level. Experiments are easy to design,
setup, and run. Existing methods are easy to modify and extend.

### Stream learning tools
In its current state, scikit-multiflow contains data generators, multi-output/multi-target stream
learning methods, change detection methods, evaluation methods, and more.

### Open source
Distributed under the
[BSD 3-Clause](https://github.com/scikit-multiflow/scikit-multiflow/blob/master/LICENSE),
`scikit-multiflow` is developed and maintained by an active, diverse and growing [community](/community).

# Use cases
The following tasks are supported in `scikit-multiflow`:

### Supervised learning
When working with labeled data. Depending on the target type can be either classification
(discrete values) or regression (continuous values)

### Single/multi output
Single-output methods predict a single target-label (binary or multi-class) for classification or
a single target-value for regression. Multi-output methods simultaneously predict multiple
variables given an input.

### Concept drift detection
Changes in data distribution can harm learning. Drift detection methods are designed to rise an
alarm in the presence of drift and are used alongside learning methods to improve their robustness
against this phenomenon in evolving data streams.

### Unsupervised learning
When working with unlabeled data. For example, anomaly detection where the goal is the
identification of rare events or samples which differ significantly from the majority of the data.

---

#### Jupyter Notebooks
In order to display plots from `scikit-multiflow` within a [Jupyter Notebook]() we need to define
the proper mathplotlib backend to use. This is done by including the following magic command at the
beginning of the Notebook:

```python
%matplotlib notebook
```

[JupyterLab](http://jupyterlab.readthedocs.io/en/stable/) is the next-generation user interface
for Jupyter, currently in beta, it can display interactive plots with some caveats. If you use
JupyterLab then the current solution is to use the
[jupyter-matplotlib](https://github.com/matplotlib/jupyter-matplotlib) extension:

```python
%matplotlib widget
```

## Citing `scikit-multiflow`

If `scikit-multiflow` has been useful for your research and you would like to cite it in a academic
publication, please use the following Bibtex entry:

```bibtex
@article{skmultiflow,
  author  = {Jacob Montiel and Jesse Read and Albert Bifet and Talel Abdessalem},
  title   = {Scikit-Multiflow: A Multi-output Streaming Framework },
  journal = {Journal of Machine Learning Research},
  year    = {2018},
  volume  = {19},
  number  = {72},
  pages   = {1-5},
  url     = {http://jmlr.org/papers/v19/18-251.html}
}
```
