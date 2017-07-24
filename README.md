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

### TODO
* (DONE) FileStream -> last_class instead of bool make it index based
* (DONE) Target instead of class
* Filters: joining, noise, remove, add
* (DONE) Plot for regression (signal vs predicted)
* (DONE) MSE and MAE
* (DONE) In the evaluation keep track of statistics instead of all the true labels and predictions
    * (DONE) Use BasicClassificationPerformanceEvaluator (it's in MOA) for this purpose. (WindowClassificationPerformanceEvaluator)
* (DONE) MultiOutputMeasurements matrix is not being filled with values... find out why
* (DONE) Work on the FastComplexBuffer, make everything work

* (DONE) MSE and MAE plotting may have some problems, because it's always really close to 0.
* (DONE, but maybe it's not the best way to plot this. Verify with Jesse) Need to check if true_vs_predict plot works
* Feature to compare multiple classifiers in the same plot

* (DONE) moa/classifiers/core/driftdetection

* (DONE) Find out why in knn lines 90-100 the new_ind (for new_indexes) the new indexes come in a list of list and not simply a list

* (DONE) I may be doing something wrong. When using adwin to manage window size, is there no window_max_size, because the algorithm is a little slow
* (DONE) Pipeline has a comment on validate_steps
* (DONE) Better accuracy on fixed size windows
* (DONE) In knn with adwin, what to feed adwin in the first k samples given to partial_fit

* (DONE) oza bagging, leverage bagging
* (DONE) finish working on knn wiht adwin

* (DONE) Not all learners in OzaBagging receive pre train because of the k variable drawn from a poisson distribution

* (DONE) What is wrong with the code matrix in leverage_bagging
* (DONE) Leverage bagging test not running correctly    -> (DONE) problem in knn.predict_proba
                                                        -> (DONE) Run simulation, something is wrong in window.add_element from knn

* Right now the nodes don't keep the samples, but maybe it will be necessary to keep them so that we can later calculate de distances
* Just noticed that assign a variable to a slice of a list actually is just a reference, so the memory cost won't be so bad
* Why in knn with k=8, the algorithm only finds the 7 nearest neighbors?????
