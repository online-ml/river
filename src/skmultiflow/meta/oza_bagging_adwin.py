import copy as cp

from skmultiflow.meta import OzaBagging
from skmultiflow.lazy import KNNAdwin
from skmultiflow.drift_detection import ADWIN
from skmultiflow.utils.utils import *


class OzaBaggingAdwin(OzaBagging):
    """ Oza Bagging ensemble classifier with ADWIN change detector.

    Parameters
    ----------
    base_estimator: skmultiflow.core.BaseSKMObject or sklearn.BaseEstimator (default=KNNAdwin)
        Each member of the ensemble is an instance of the base estimator.

    n_estimators: int (default=10)
        The size of the ensemble, in other words, how many classifiers to train.

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by `np.random`.
    
    Raises
    ------
    ValueError: A ValueError is raised if the 'classes' parameter is
    not passed in the first partial_fit call.

    Notes
    -----
    This online ensemble learner method is an improvement from the Online
    Bagging algorithm [1]_. The improvement comes from the addition of a ADWIN change
    detector.

    ADWIN stands for Adaptive Windowing. It works by keeping updated
    statistics of a variable sized window, so it can detect changes and
    perform cuts in its window to better adapt the learning algorithms.

    References
    ----------
    .. [1] N. C. Oza, “Online Bagging and Boosting,” in 2005 IEEE International Conference on Systems,
       Man and Cybernetics, 2005, vol. 3, no. 3, pp. 2340–2345.

    
    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.meta import OzaBaggingAdwin
    >>> from skmultiflow.lazy.knn import KNN
    >>> from skmultiflow.data.sea_generator import SEAGenerator
    >>> # Setting up the stream
    >>> stream = SEAGenerator(1, noise_percentage=6.7)
    >>> stream.prepare_for_use()
    >>> # Setting up the OzaBagginAdwin classifier to work with KNN classifiers
    >>> clf = OzaBaggingAdwin(base_estimator=KNN(n_neighbors=8, max_window_size=2000, leaf_size=30), n_estimators=2)
    >>> # Keeping track of sample count and correct prediction count
    >>> sample_count = 0
    >>> corrects = 0
    >>> # Pre training the classifier with 200 samples
    >>> X, y = stream.next_sample(200)
    >>> clf = clf.partial_fit(X, y, classes=stream.target_values)
    >>> for i in range(2000):
    ...     X, y = stream.next_sample()
    ...     pred = clf.predict(X)
    ...     clf = clf.partial_fit(X, y)
    ...     if pred is not None:
    ...         if y[0] == pred[0]:
    ...             corrects += 1
    ...     sample_count += 1
    >>> 
    >>> # Displaying the results
    >>> print(str(sample_count) + ' samples analyzed.')
    2000 samples analyzed.
    >>> print('OzaBaggingAdwin classifier performance: ' + str(corrects / sample_count))
    OzaBaggingAdwin classifier performance: 0.9645
    
    """

    def __init__(self, base_estimator=KNNAdwin(), n_estimators=10, random_state=None):
        super().__init__(base_estimator, n_estimators, random_state)
        # default values
        self.adwin_ensemble = None
        self.__configure()

    def __configure(self):
        self.adwin_ensemble = [cp.deepcopy(ADWIN()) for _ in range(self.actual_n_estimators)]

    def reset(self):
        self.__configure()
        return self

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """ Partially (incrementally) fit the model.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.

        y: numpy.ndarray of shape (n_samples)
            An array-like with the class labels of all samples in X.

        classes: numpy.ndarray, optional (default=None)
            Array with all possible/known class labels. This is an optional parameter, except
            for the first partial_fit call where it is compulsory.

        sample_weight: numpy.ndarray of shape (n_samples), optional (default=None)
            Samples weight. If not provided, uniform weights are assumed. Usage varies depending on the base estimator.

        Raises
        ------
        ValueError: A ValueError is raised if the 'classes' parameter is not
        passed in the first partial_fit call, or if they are passed in further 
        calls but differ from the initial classes list passed.

        Returns
        _______
        OzaBaggingAdwin
            self

        Notes
        -----
        Since it's an ensemble learner, if X and y matrix of more than one
        sample are passed, the algorithm will partial fit the model one sample
        at a time.

        Each sample is trained by each classifier a total of K times, where K
        is drawn by a Poisson(1) distribution.

        Alongside updating the model, the learner will also update ADWIN's
        statistics over the new samples, so that the change detector can
        evaluate if a concept drift was detected. In the case drift is detected,
        the bagging algorithm will find the worst performing classifier and reset
        its statistics and window.

        """
        r, c = get_dimensions(X)
        if self.classes is None:
            if classes is None:
                raise ValueError("The first partial_fit call should pass all the classes.")
            else:
                self.classes = classes

        if self.classes is not None and classes is not None:
            if set(self.classes) == set(classes):
                pass
            else:
                raise ValueError(
                    "The classes passed to the partial_fit function differ from those passed in an earlier moment.")

        self.__adjust_ensemble_size()
        change_detected = False
        for i in range(self.actual_n_estimators):
            k = self._random_state.poisson()
            if k > 0:
                for b in range(k):
                    self.ensemble[i].partial_fit(X, y, classes, sample_weight)

            try:
                pred = self.ensemble[i].predict(X)
                error_estimation = self.adwin_ensemble[i].estimation
                for j in range(r):
                    if pred[j] is not None:
                        if pred[j] == y[j]:
                            self.adwin_ensemble[i].add_element(1)
                        else:
                            self.adwin_ensemble[i].add_element(0)
                if self.adwin_ensemble[i].detected_change():
                    if self.adwin_ensemble[i].estimation > error_estimation:
                        change_detected = True
            except ValueError:
                change_detected = False
                pass

        if change_detected:
            max_threshold = 0.0
            i_max = -1
            for i in range(self.actual_n_estimators):
                if max_threshold < self.adwin_ensemble[i].estimation:
                    max_threshold = self.adwin_ensemble[i].estimation
                    i_max = i
            if i_max != -1:
                self.ensemble[i_max].reset()
                self.adwin_ensemble[i_max] = ADWIN()

        return self

    def __adjust_ensemble_size(self):
        if len(self.classes) != len(self.ensemble):
            if len(self.classes) > len(self.ensemble):
                for i in range(len(self.ensemble), len(self.classes)):
                    self.ensemble.append(cp.deepcopy(self.base_estimator))
                    self.adwin_ensemble.append(ADWIN())
                    self.actual_n_estimators += 1
