For developers
==============

Code style and documentation
----------------------------

* Python Code shall comply with `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_

* Documentation shall be in docstring format and shall follow the
  `NumPy/SciPy guidelines <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_

  - An example from the sphinx documentation:

    http://www.sphinx-doc.org/en/stable/ext/example_numpy.html

Independent demos
-----------------

To run independent demos, make your way in a terminal to ".../skmultiflow/demos/" and run:

::

  python demo_name.py


matplotlib backend considerations
---------------------------------

* You may need to change your matplotlib backend, because not all backends work in all machines
* If this is the case and you use Conda, you can find the virtual environment matplotlib config file in::

    /miniconda3/pkgs/matplotlib-2.0.2-np112py35_0/lib/python3.5/site-packages/matplotlib/mpl-data/matplotlibrc

* In the matplotlibrc file you will need to change the line:
    backend     : Qt5Agg

    to:

    backend     : another backend that works on your machine

* The ``Qt5Agg`` backend should work with most machines, but a change may be needed.

Sphinx documentation
--------------------

* We generate our documentation directly through sphinx.
* To update the documentation perform in a terminal the steps below:
    * Make your way to the scikit-multiflow parent directory (outside the project)
    * Type in::

        sphinx-apidoc -o scikit-multiflow/docs scikit-multiflow/ -e -f

    * Go to scikit-multiflow/docs
    * Type in::

        make html

License
-------
* 3-Clause BSD License
