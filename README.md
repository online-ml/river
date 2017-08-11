# scikit-multiflow

A stream data framework.

### Dependencies

* python3
* matplotlib
* numpy
* scipy
* pandas
* scikit-learn

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

### Terminal library example
* Go to skmultiflow root folder
* Open a python terminal
* Type:
    >>> from skmultiflow.data.FileStream import FileStream
    >>> from skmultiflow.options.FileOption import FileOption
    >>> from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier
    >>> from skmultiflow.core.pipeline.Pipeline import Pipeline
    >>> from skmultiflow.evaluation.EvaluatePrequential import EvaluatePrequential
    >>> opt = FileOption('FILE', 'OPT_NAME', 'skmultiflow/datasets/covtype.csv', 'CSV', False)
    >>> stream = FileStream(opt, 7)
    >>> stream.prepare_for_use()
    >>> classifier = PassiveAggressiveClassifier()
    >>> pipe = Pipeline([('Pipeline', classifier)])
    >>> eval = EvaluatePrequential(show_plot=True, pretrain_size=1000)
    >>> eval.eval(stream=stream, classifier=pipe)
* If your system doesn't support a graphic interface change the line:
    >>> eval = EvaluatePrequential(show_plot=True, pretrain_size=1000)
    to:
    >>> eval = EvaluatePrequential(show_plot=False, pretrain_size=1000)

### Terminal demos
* Go to skmultiflow root folder
* Enter python testing.py
* Let the demo run
* To change the demo you want simply comment/uncomment the testing.py lines, in the root directory

### matplotlib backend considerations
* You may need to change your matplotlib backend, because not all backends work in all machines
* If this is the case you can find the conda virtual environment matplotlib config file in:
    /miniconda3/pkgs/matplotlib-2.0.2-np112py35_0/lib/python3.5/site-packages/matplotlib/mpl-data/matplotlibrc
* In the matplotlibrc file you will need to change the line:
    backend     : macosx
    to:
    backend     : another backend that works on your machine

### To clean
* core.instances

* Cython
* Numba

* stream_classifier
* test regression