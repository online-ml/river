import copy as cp

from skmultiflow.core.base import BaseSKMObject, ClassifierMixin, MetaEstimatorMixin
from skmultiflow.lazy import KNNClassifier
from river.drift import ADWIN
from skmultiflow.utils.utils import *
from skmultiflow.utils import check_random_state

import warnings


def LeverageBagging(base_estimator=KNNClassifier(), n_estimators=10, w=6, delta=0.002,
                    enable_code_matrix=False, leverage_algorithm='leveraging_bag',
                    random_state=None):     # pragma: no cover
    warnings.warn("'LeverageBagging' has been renamed to 'LeveragingBaggingClassifier' in "
                  "v0.5.0.\nThe old name will be removed in v0.7.0", category=FutureWarning)
    return LeveragingBaggingClassifier(base_estimator=base_estimator,
                                       n_estimators=n_estimators,
                                       w=w,
                                       delta=delta,
                                       enable_code_matrix=enable_code_matrix,
                                       leverage_algorithm=leverage_algorithm,
                                       random_state=random_state)


class LeveragingBaggingClassifier(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):
    """ Leveraging Bagging ensemble classifier.

    Parameters
    ----------
    base_estimator: skmultiflow.core.BaseSKMObject or sklearn.BaseEstimator \
    (default=KNN)
        Each member of the ensemble is an instance of the base estimator.

    n_estimators: int (default=10)
        The size of the ensemble, in other words, how many classifiers to train.

    w: int (default=6)
        The poisson distribution's parameter, which is used to simulate
        re-sampling.

    delta: float (default=0.002)
        The delta parameter for the ADWIN change detector.

    enable_code_matrix: bool (default=False)
        If set, enables Leveraging Bagging MC using Random Output Codes.

    leverage_algorithm: string (default='leveraging_bag')
        | The bagging algorithm to use. Can be one of the following:
        | 'leveraging_bag' - Leveraging Bagging using ADWIN
        | 'leveraging_bag_me' - Assigns to a sample ``weight=1`` if \
        misclassified, otherwise ``weight=error/(1-error)``
        | 'leveraging_bag_half' - Use resampling without replacement for half \
        of the instances
        | 'leveraging_bag_wt' - Without taking out all instances
        | 'leveraging_subag' - Using resampling without replacement

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Raises
    ------
    ValueError: A ValueError is raised if the ``classes`` parameter is not \
    passed in the first ``partial_fit`` call.

    Notes
    -----
    An ensemble method, which represents an improvement from the online Oza
    Bagging algorithm. The complete description of this method can be found
    in [1]_.

    The bagging performance is leveraged by increasing the re-sampling and
    by using output detection codes. We use a poisson distribution to
    simulate the re-sampling process. To increase re-sampling we use a higher
    value of the w parameter of the Poisson distribution, which is 6 by
    default. With this value we are increasing the input space diversity, by
    attributing a different range of weights to our samples.

    The second improvement is to use output detection codes. This consists of
    coding each label with a n bit long binary code and then associating n
    classifiers, one to each bit of the binary codes. At each new sample
    analyzed, each classifier is trained on its own bit. This allows, to some
    extent, the correction of errors.

    To deal with concept drift we use the ADWIN algorithm, one instance for
    each classifier. Each time a concept drift is detected we reset the worst
    ensemble's classifier, which is done by comparing the adwins' window sizes.

    References
    ----------
    .. [1] A. Bifet, G. Holmes, and B. Pfahringer, “Leveraging Bagging for
       Evolving Data Streams,” in Joint European Conference on Machine Learning
       and Knowledge Discovery in Databases, 2010, no. 1, pp. 135–150.


    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.meta import LeveragingBaggingClassifier
    >>> from skmultiflow.lazy import KNNClassifier
    >>> from skmultiflow.data import SEAGenerator
    >>> # Setting up the stream
    >>> stream = SEAGenerator(1, noise_percentage=.067)
    >>> # Setting up the LeverageBagging classifier to work with KNN classifiers
    >>> clf = LeveragingBaggingClassifier(base_estimator=
    >>>                                 KNNClassifier(n_neighbors=8,
    >>>                                               max_window_size=2000,
    >>>                                               leaf_size=30)
    >>>                                 , n_estimators=2)
    >>> # Keeping track of sample count and correct prediction count
    >>> sample_count = 0
    >>> corrects = 0
    >>> for i in range(2000):
    >>>     X, y = stream.next_sample()
    >>>     pred = clf.predict(X)
    >>>     clf = clf.partial_fit(X, y, classes=stream.target_values)
    >>>     if pred is not None:
    >>>         if y[0] == pred[0]:
    ...             corrects += 1
    ...     sample_count += 1
    ... # Displaying the results
    ... print(str(sample_count) + ' samples analyzed.')
    2000 samples analyzed.
    >>> print('LeveragingBaggingClassifier performance: ' + str(corrects / sample_count))
    LeveragingBagging classifier performance: 0.843

    """

    _LEVERAGE_ALGORITHMS = ['leveraging_bag',
                            'leveraging_bag_me',
                            'leveraging_bag_half',
                            'leveraging_bag_wt',
                            'leveraging_subag']

    def __init__(self,
                 base_estimator=KNNClassifier(),
                 n_estimators=10,
                 w=6,
                 delta=0.002,
                 enable_code_matrix=False,
                 leverage_algorithm='leveraging_bag',
                 random_state=None):

        super().__init__()
        # default values
        self.ensemble = None
        self.adwin_ensemble = None
        self.n_detected_changes = None
        self.matrix_codes = None
        self.classes = None
        self.init_matrix_codes = None
        self._random_state = None   # This is the actual random_state object used internally
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.enable_code_matrix = enable_code_matrix
        self.w = w
        self.delta = delta
        if leverage_algorithm not in self._LEVERAGE_ALGORITHMS:
            raise ValueError("Invalid option for leverage_algorithm: '{}'\n"
                             "Valid options are: {}".format(leverage_algorithm,
                                                            self._LEVERAGE_ALGORITHMS))
        self.leverage_algorithm = leverage_algorithm
        self.random_state = random_state
        self.__configure()

    def __configure(self):
        if hasattr(self.base_estimator, "reset"):
            self.base_estimator.reset()
        self.actual_n_estimators = self.n_estimators
        self.ensemble = [cp.deepcopy(self.base_estimator) for _ in range(self.actual_n_estimators)]
        self.adwin_ensemble = [ADWIN(self.delta) for _ in range(self.actual_n_estimators)]
        self._random_state = check_random_state(self.random_state)
        self.n_detected_changes = 0
        self.classes = None
        self.init_matrix_codes = True

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """ Partially (incrementally) fit the model.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.

        y: numpy.ndarray of shape (n_samples)
            An array-like with the class labels of all samples in X.

        classes: numpy.ndarray, optional (default=None)
            Array with all possible/known class labels.

        sample_weight: not used (default=None)

        Raises
        ------
        ValueError: A ValueError is raised if the 'classes' parameter is not
        passed in the first partial_fit call, or if they are passed in further
        calls but differ from the initial classes list passed.

        Returns
        -------
        LeveragingBaggingClassifier
            self

        """
        if classes is None and self.classes is None:
            raise ValueError("The first partial_fit call should pass all the classes.")
        if classes is not None and self.classes is None:
            self.classes = classes
        elif classes is not None and self.classes is not None:
            if set(self.classes) == set(classes):
                pass
            else:
                raise ValueError(
                    "The classes passed to the partial_fit function differ from those passed in "
                    "an earlier call.")

        r, c = get_dimensions(X)
        for i in range(r):
            self.__partial_fit(X[i], y[i])

        return self

    def __partial_fit(self, X, y):
        if self.init_matrix_codes and self.enable_code_matrix:
            self.__init_output_codes()

        change_detected = False
        for i in range(self.actual_n_estimators):

            # leveraging_bag - Leveraging Bagging
            if self.leverage_algorithm == self._LEVERAGE_ALGORITHMS[0]:
                k = self._random_state.poisson(self.w)

            # leveraging_bag_me - Missclassification Error
            elif self.leverage_algorithm == self._LEVERAGE_ALGORITHMS[1]:
                error = self.adwin_ensemble[i].estimation
                pred = self.ensemble[i].predict(np.asarray([X]))
                if pred is None:
                    k = 1.0
                elif pred[0] != y:
                    k = 1.0
                elif (error != 1.0 and
                      self._random_state.rand() < (error / (1.0 - error))):
                    k = 1.0
                else:
                    k = 0.0

            # leveraging_bag_half - Resampling without replacement for
            #                       half of the instances
            elif self.leverage_algorithm == self._LEVERAGE_ALGORITHMS[2]:
                w = 1.0
                k = 0.0 if (self._random_state.randint(2) == 1) else w

            # leveraging_bag_wt - Without taking out all instances
            elif self.leverage_algorithm == self._LEVERAGE_ALGORITHMS[3]:
                w = 1.0
                k = 1.0 + self._random_state.poisson(w)

            # leveraging_subag - Resampling without replacement
            elif self.leverage_algorithm == self._LEVERAGE_ALGORITHMS[4]:
                w = 1.0
                k = self._random_state.poisson(1)
                k = w if k > 0 else 0

            else:
                raise RuntimeError("Invalid option for leverage_algorithm: '{}'\n"
                                   "Valid options are: {}".format(self.leverage_algorithm,
                                                                  self._LEVERAGE_ALGORITHMS))

            y_coded = cp.deepcopy(y)
            if k > 0:
                classes = self.classes
                if self.enable_code_matrix:
                    y_coded = self.matrix_codes[i][int(y)]
                    classes = [0, 1]
                for _ in range(int(k)):
                    self.ensemble[i].partial_fit(X=np.asarray([X]), y=np.asarray([y_coded]),
                                                 classes=classes)

            pred = self.ensemble[i].predict(np.asarray([X]))
            if pred is not None:
                add = 0 if (pred[0] == y_coded) else 1
                error = self.adwin_ensemble[i].estimation
                self.adwin_ensemble[i].update(add)
                if self.adwin_ensemble[i].change_detected:
                    if self.adwin_ensemble[i].estimation > error:
                        change_detected = True

        if change_detected:
            self.n_detected_changes += 1
            max_threshold = 0.0
            i_max = -1
            for i in range(self.actual_n_estimators):
                if max_threshold < self.adwin_ensemble[i].estimation:
                    max_threshold = self.adwin_ensemble[i].estimation
                    i_max = i
            if i_max != -1:
                self.ensemble[i_max].reset()
                self.adwin_ensemble[i_max] = ADWIN(self.delta)
        return self

    def __init_output_codes(self):
        self.matrix_codes = np.zeros((self.actual_n_estimators, len(self.classes)), dtype=int)
        for i in range(self.actual_n_estimators):
            condition = True
            while condition:
                n_zeros = 0
                n_ones = 0
                for j in range(len(self.classes)):
                    if (j == 1) and (len(self.classes) == 2):
                        result = 1 - self.matrix_codes[i][0]
                    else:
                        result = self._random_state.randint(2)

                    self.matrix_codes[i][j] = result
                    if result == 1:
                        n_ones += 1
                    else:
                        n_zeros += 1
                condition = ((n_ones - n_zeros) * (n_ones - n_zeros) >
                             (self.actual_n_estimators % 2))
        self.init_matrix_codes = False

    def predict(self, X):
        """ Predict classes for the passed data.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The set of data samples to predict the class labels for.

        Returns
        -------
        A numpy.ndarray with all the predictions for the samples in X.

        """
        r, c = get_dimensions(X)
        proba = self.predict_proba(X)
        predictions = []
        if proba is None:
            # Ensemble is empty, all classes equal, default to zero
            proba = np.zeros((r, 1))
        for i in range(r):
            predictions.append(np.argmax(proba[i]))
        return np.asarray(predictions)

    def predict_proba(self, X):
        """ Estimate the probability of X belonging to each class-label.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The matrix of samples to predict the class probabilities for.

        Raises
        ------
        ValueError: A ValueError is raised if the number of classes in the base
        learner exceed that of the ensemble learner.

        Returns
        -------
        A numpy.ndarray of shape (n_samples, n_labels), in which each outer
        entry is associated with the X entry of the same index. And where the
        list in index [i] contains len(self.target_values) elements, each of
        which represents the probability that the i-th sample of X belongs to
        a certain class-label.

        Notes
        -----
        Calculates the probability of each sample in X belonging to each
        of the labels, based on the base estimator. This is done by predicting
        the class probability for each one of the ensemble's classifier, and
        then taking the absolute probability from the ensemble itself.

        """
        if self.enable_code_matrix:
            return self.predict_binary_proba(X)
        proba = []
        r, c = get_dimensions(X)
        try:
            for i in range(self.actual_n_estimators):
                partial_proba = self.ensemble[i].predict_proba(X)
                if len(partial_proba[0]) > max(self.classes) + 1:
                    raise ValueError("The number of classes in the base learner is larger than in"
                                     " the ensemble.")

                if len(proba) < 1:
                    for row_idx in range(r):
                        proba.append([0.0] * len(partial_proba[row_idx]))

                for row_idx in range(r):
                    for class_idx in range(len(partial_proba[row_idx])):
                        try:
                            proba[row_idx][class_idx] += partial_proba[row_idx][class_idx]
                        except IndexError:
                            proba[row_idx].append(partial_proba[row_idx][class_idx])
        except ValueError:
            return np.zeros((r, 1))
        except TypeError:
            return np.zeros((r, 1))

        return self._normalize_probabilities(rows=r, y_proba=proba)

    def predict_binary_proba(self, X):
        """ Calculates the probability of a sample belonging to each coded label.

        This will only be used if matrix codes are enabled.
        Otherwise the method will use the normal predict_proba function.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            All the samples we want to predict the label for.

        Returns
        -------
        list
            A list of lists, in which each outer entry is associated with
            the X entry of the same index. And where the list in index [i]
            contains len(self.target_values) elements, each of which represents
            the probability that the i-th sample of X belongs to a certain
            label.

        """
        r, c = get_dimensions(X)
        if self.classes is None:
            return np.zeros(r)
        proba = np.zeros((r, len(self.classes)))
        if not self.init_matrix_codes:
            for row_idx in range(r):
                for i in range(self.actual_n_estimators):
                    vote = self.ensemble[i].predict_proba(X)
                    vote_class = 0
                    if len(vote[row_idx]) == 2:
                        vote_class = 1 if (vote[row_idx][1] > vote[row_idx][0]) else 0

                    for j in range(len(self.classes)):
                        if self.matrix_codes[i][j] == vote_class:
                            proba[row_idx][j] += 1
            return self._normalize_probabilities(rows=r, y_proba=proba)
        return proba

    @staticmethod
    def _normalize_probabilities(rows: int, y_proba):
        # normalizing probabilities
        sum_proba = []
        for row in range(rows):
            sum_proba.append(np.sum(y_proba[row]))
        aux = []
        for i in range(len(y_proba)):
            if sum_proba[i] > 0.:
                aux.append([x / sum_proba[i] for x in y_proba[i]])
            else:
                aux.append(y_proba[i])
        return np.asarray(aux)

    def reset(self):
        """ Resets all the estimators, as well as all the ADWIN change detectors.

        Returns
        -------
        LeveragingBaggingClassifier
            self
        """
        self.__configure()
        return self
