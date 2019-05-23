<img src="https://scikit-multiflow.github.io/scikit-multiflow/_images/skmultiflow-logo-wide.png" height="100"/>

[![Build status](https://travis-ci.org/scikit-multiflow/scikit-multiflow.svg?branch=master)](https://travis-ci.org/scikit-multiflow/scikit-multiflow)
[![codecov](https://codecov.io/gh/scikit-multiflow/scikit-multiflow/branch/master/graph/badge.svg)](https://codecov.io/gh/scikit-multiflow/scikit-multiflow)
![Python version](https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7-blue.svg)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/scikit-multiflow/badges/platforms.svg)](https://anaconda.org/conda-forge/scikit-multiflow)
[![PyPI version](https://badge.fury.io/py/scikit-multiflow.svg)](https://badge.fury.io/py/scikit-multiflow)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/scikit-multiflow/badges/version.svg)](https://anaconda.org/conda-forge/scikit-multiflow)
[![DockerHub](https://img.shields.io/badge/docker-available-blue.svg?logo=docker)](https://hub.docker.com/r/skmultiflow/scikit-multiflow)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Gitter](https://badges.gitter.im/scikit-multiflow/community.svg)](https://gitter.im/scikit-multiflow/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

A machine learning framework for multi-output/multi-label and stream data.
Inspired by [MOA](https://moa.cms.waikato.ac.nz/) and [MEKA](http://meka.sourceforge.net/),
 following [scikit-learn](http://scikit-learn.org/stable/)'s philosophy.

* [Webpage](https://scikit-multiflow.github.io/)
* [Documentation](https://scikit-multiflow.github.io/scikit-multiflow/)
* [Users Group](https://groups.google.com/forum/#!forum/scikit-multiflow-users)

### matplotlib backend considerations
* You may need to change your matplotlib backend, because not all backends work
in all machines.
* If this is the case you need to check
[matplotlib's configuration](https://matplotlib.org/users/customizing.html).
In the matplotlibrc file you will need to change the line:  
    ```
    backend     : Qt5Agg  
    ```
    to:  
    ```
    backend     : another backend that works on your machine
    ```  
* The Qt5Agg backend should work with most machines, but a change may be needed.

#### Jupyter Notebooks
In order to display plots from `scikit-multiflow` within a [Jupyter Notebook]() we need to define the proper mathplotlib
backend to use. This is done via a magic command at the beginning of the Notebook:

```python
%matplotlib notebook
```

[JupyterLab](http://jupyterlab.readthedocs.io/en/stable/) is the next-generation user interface for Jupyter, currently
in beta, it can display interactive plots with some caveats. If you use JupyterLab then the current solution is to use
the [jupyter-matplotlib](https://github.com/matplotlib/jupyter-matplotlib) extension:

```python
%matplotlib widget
```

### Citing `scikit-multiflow`

If you want to cite `scikit-multiflow` in a scientific publication, please use the following Bibtex entry:
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
