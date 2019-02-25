============
Installation
============

**Notice:** `scikit-multiflow` works with Python 3.4+ **only**.

``scikit-multiflow`` requires `numpy <www.numpy.org>`_ to be already installed in your system. There are multiple ways to install ``numpy``, the easiest is using `pip <https://pip.pypa.io/en/stable/#>`_:

.. code-block:: bash

   $ pip install -U numpy

Option 1. Install from source code
==================================

First, you need to make a copy of the ``scikit-multiflow`` project. On the `project's github page <https://github.com/scikit-multiflow/scikit-multiflow>`_ you will find on the top-right side of the page a green button with the label "Clone or download". By clicking on it you will get two options: clone with SSH, HTTPS or download a zip. If you opt to get a zip file then you have to unzip the project into the desired local destination before continuing.

Once `numpy` is installed, you can proceed with the installation of ``scikit-multiflow`` and its other dependencies.

In a terminal, navigate to the local path of the project and run the following command (including the dot at the end):

.. code-block:: bash

   $ pip install -U .

The `-U` option indicates that the package will be installed only for this user.

Optionally you can indicate to `pip` the remote location of the code:

.. code-block:: bash

   $ pip install -U git+https://github.com/scikit-multiflow/scikit-multiflow

When the installation is completed (and no errors were reported), then you will be ready to use `scikit-multiflow`. The advantage of this option is that you can install the latest version of the code in github.

Option 2. Install from PyPI
===========================

`scikit-multiflow` is also available in the Python Package Index (PyPI). So you can install it using the following command:

.. code-block:: bash

   $ pip install -U scikit-multiflow

This will install the latest (stable) release of the code.

Option 3. Install with Docker
=============================
`scikit-multiflow` is also available in Docker. Docker images are located in the `skmultiflow/scikit-multiflow <https://hub.docker.com/r/skmultiflow/scikit-multiflow>`_ Docker Hub repository.

You can download the image and start using `scikit-multiflow`.

Download `scikit-multiflow` Docker image

.. code-block:: bash

    $ docker pull skmultiflow/scikit-multiflow:latest

Start `scikit-multiflow` Docker container

.. code-block:: bash

    $ docker run -it skmultiflow/scikit-multiflow:latest

Run the Hoeffding Tree example

.. code-block:: bash

    $ python hoeffding_tree.py


Also, we provide images with `Jupyter Notebook <http://jupyter.org/>`_. For more details, see `skmultiflow/scikit-multiflow <https://hub.docker.com/r/skmultiflow/scikit-multiflow>`_ Docker Hub repository.

Option 4. Development version
====================================

For people interested in contributing to `scikit-multiflow` we recommend to install the project in *editable* mode, please refer to the `contributor's page <https://github.com/scikit-multiflow/scikit-multiflow/blob/master/CONTRIBUTING.md>`_ for further information.


matplotlib backend considerations
=================================

* You may need to change your matplotlib backend, because not all backends work on all machines.
* If this is the case you need to check  `matplotlib's configuration <https://matplotlib.org/users/customizing.html>`_. In the *matplotlibrc* file you will need to change the line:

  ::

   backend     : Qt5Agg

  to:

  ::

   backend     : a backend that works on your machine


* The **Qt5Agg** backend should work with most machines, but a change may be needed.

Jupyter Notebooks
=================

In order to display plots from ``scikit-multiflow`` within a `Jupyter Notebook <http://jupyter.org/>`_ we need to define the proper ``mathplotlib`` backend to use. This is done via a magic command at the beginning of the Notebook:

.. code-block:: python

   %matplotlib notebook


`JupyterLab <http://jupyterlab.readthedocs.io/en/stable/>`_ is Jupyter's *next-generation* user interface, currently in beta it can display plots with some caveats. If you use JupyterLab then the current solution is to use the `jupyter-matplotlib <https://github.com/matplotlib/jupyter-matplotlib>`_ extension:

.. code-block:: python

   %matplotlib ipympl

