# scikit-multiflow

A stream data framework.

### Dependencies
* python3
* matplotlib
* numpy
* scipy
* pandas
* scikit-learn

* Due to a known issue with NumPy's install requirements, all of the dependencies should be manually installed.
Then the setup.py can be run to install the scikit-multiflow package.

### Project leaders

* Albert BIFET
* Jesse READ
* Jacob MONTIEL

### Contributors

* Guilherme KURIKE MATSUMOTO


### Code style and documentation
* Python Code shall comply with [PEP 8](https://www.python.org/dev/peps/pep-0008/)

* Documentation shall be in docstring format and shall follow the [NumPy/SciPy guidelines](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt)

    An example from the sphinx documentation: http://www.sphinx-doc.org/en/stable/ext/example_numpy.html

### Testing considerations
* Always be in the correct virtual environment.
* I'll send you through the google hangouts chat the conda configuration file
* See the matplotlib backend considerations in the end of this README.md file

### Independent demos
To run independent demos, make your way to ".../skmultiflow/demos/" and run in a terminal:
>>> python <test_name>

### matplotlib backend considerations
* You may need to change your matplotlib backend, because not all backends work in all machines
* If this is the case you can find the conda virtual environment matplotlib config file in:
    /miniconda3/pkgs/matplotlib-2.0.2-np112py35_0/lib/python3.5/site-packages/matplotlib/mpl-data/matplotlibrc
* In the matplotlibrc file you will need to change the line:
    backend     : Qt5Agg
    to:
    backend     : another backend that works on your machine
* The Qt5Agg backend should work with most machines, but a change may be needed.

### License
* 3-Clause BSD License

* SAM knn
* classifiers to regressor in visualization

