# scikit-multiflow

A stream data framework.

### Dependencies

* python3
* matplotlib
* numpy

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

### TODO
* Fully implement the StreamCreator function and the ClassifierCreator function
    * Verify that no outter shell will be needed for the classifiers coming directly from sklearn -> maybe do additional
     treatment in the creator functions
* Continue working in the parsing functions
* Implement at least one classifier for testing purposes
* Finish Prequential Evaluator