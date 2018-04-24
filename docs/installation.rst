============
Installation
============

**Notice:** `scikit-multiflow` works with Python 3.4+ **only**.

First, you need to make a copy of the ``scikit-multiflow`` project. On the `project's github page <https://github.com/scikit-multiflow/scikit-multiflow>`_ you will find on the top-right side of the page a green button with the label "Clone or download". By clicking on it you will get two options: clone with SSH or download a zip. If you opt to get a zip file then you have to unzip the project into the desired local destination before continuing.

In a terminal, navigate to the local path of the project.

``scikit-multiflow`` requires `numpy <www.numpy.org>`_ to be already installed in your system. There are multiple ways to install ``numpy``, the easiest is using `pip <https://pip.pypa.io/en/stable/#>`_:

.. code-block:: bash

   $ pip install -U numpy


Once `numpy` is installed, you can proceed with the installation of ``scikit-multiflow`` and its other dependencies. Run the following command (including the dot at the end):

.. code-block:: bash

   $ pip install -U .


When the installation is completed (and no errors were reported), then you will be ready to use `scikit-multiflow`.

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

