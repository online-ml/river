import copy as cp

from skmultiflow.core.base import BaseSKMObject, ClassifierMixin, MetaEstimatorMixin
from skmultiflow.lazy import KNN
from skmultiflow.drift_detection import ADWIN
from skmultiflow.utils.utils import *
from skmultiflow.utils import check_random_state


class LeverageBagging(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):
    """ Leverage Bagging ensemble classifier.

    Parameters
    ----------
    base_estimator: skmultiflow.core.BaseSKMObject or sklearn.BaseEstimator (default=KNN)
        Each member of the ensemble is an instance of the base estimator.
        
    n_estimators: int (default=10)
        The size of the ensemble, in other words, how many classifiers to train.

    w: int (default=6)
        The poisson distribution's parameter, which is used to simulate re-sampling.
        
    delta: float (default=0.002)
        The delta parameter for the ADWIN change detector.
    
    enable_code_matrix: bool (default=False)
        If set, it will enable the output detection code matrix.
    
    leverage_algorithm: string (default='leveraging_bag')
        The bagging algorithm to use. Can be one of the following: 'leveraging_bag',
         'leveraging_bag_me', 'leveraging_bag_half', 'leveraging_bag_wt', 'leveraging_subag'

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
    .. [1] A. Bifet, G. Holmes, and B. Pfahringer, “Leveraging Bagging for Evolving Data Streams,”
       in Joint European conference on machine learning and knowledge discovery in databases, 2010,
       no. 1, pp. 135–150.

    
    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.meta import LeverageBagging
    >>> from skmultiflow.lazy import KNN
    >>> from skmultiflow.data import SEAGenerator
    >>> # Setting up the stream
    >>> stream = SEAGenerator(1, noise_percentage=6.7)
    >>> stream.prepare_for_use()
    >>> # Setting up the LeverageBagging classifier to work with KNN classifiers
    >>> clf = LeverageBagging(base_estimator=KNN(n_neighbors=8, max_window_size=2000, leaf_size=30), n_estimators=2)
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
    >>> print('LeverageBagging classifier performance: ' + str(corrects / sample_count))
    LeverageBagging classifier performance: 0.945
    
    """

    LEVERAGE_ALGORITHMS = ['leveraging_bag', 'leveraging_bag_me', 'leveraging_bag_half', 'leveraging_bag_wt',
                           'leveraging_subag']

    def __init__(self,
                 base_estimator=KNN(),
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
        if leverage_algorithm not in self.LEVERAGE_ALGORITHMS:
            raise ValueError("Leverage algorithm not supported.")
        self.leverage_algorithm = leverage_algorithm
        self.random_state = random_state
        self.__configure()

    def __configure(self):
        if hasattr(self.base_estimator, "reset"):
            self.base_estimator.reset()
        self.actual_n_estimators = self.n_estimators
        self.ensemble = [cp.deepcopy(self.base_estimator) for _ in range(self.actual_n_estimators)]
        self.adwin_ensemble = []
        for i in range(self.actual_n_estimators):
            self.adwin_ensemble.append(ADWIN(self.delta))
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
        LeverageBagging
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
                    "The classes passed to the partial_fit function differ from those passed in an earlier moment.")

        r, c = get_dimensions(X)
        for i in range(r):
            self.__partial_fit(X[i], y[i])

        return self

    def __partial_fit(self, X, y):
        if self.init_matrix_codes:
            self.matrix_codes = np.zeros((self.actual_n_estimators, len(self.classes)), dtype=int)
            for i in range(self.actual_n_estimators):
                n_zeros = 0
                n_ones = 0
                while (n_ones - n_zeros) * (n_ones - n_zeros) > self.actual_n_estimators % 2:
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
            self.init_matrix_codes = False

        change_detected = False
        X_cp, y_cp = cp.deepcopy(X), cp.deepcopy(y)
        for i in range(self.actual_n_estimators):
            k = 0.0

            if self.leverage_algorithm == self.LEVERAGE_ALGORITHMS[0]:
                k = self._random_state.poisson(self.w)

            elif self.leverage_algorithm == self.LEVERAGE_ALGORITHMS[1]:
                error = self.adwin_ensemble[i].estimation
                pred = self.ensemble[i].predict(np.asarray([X]))
                if pred is None:
                    k = 1.0
                elif pred[0] != y:
                    k = 1.0
                elif self._random_state.rand() < (error / (1.0 - error)):
                    k = 1.0
                else:
                    k = 0.0

            elif self.leverage_algorithm == self.LEVERAGE_ALGORITHMS[2]:
                w = 1.0
                k = 0.0 if (self._random_state.randint(2) == 1) else w

            elif self.leverage_algorithm == self.LEVERAGE_ALGORITHMS[3]:
                w = 1.0
                k = 1.0 + self._random_state.poisson(w)

            elif self.leverage_algorithm == self.LEVERAGE_ALGORITHMS[4]:
                w = 1.0
                k = self._random_state.poisson(1)
                k = w if k > 0 else 0

            if k > 0:
                if self.enable_code_matrix:
                    y_cp = self.matrix_codes[i][int(y_cp)]
                for l in range(int(k)):
                    self.ensemble[i].partial_fit(np.asarray([X_cp]), np.asarray([y_cp]), self.classes)

            try:
                pred = self.ensemble[i].predict(np.asarray([X]))
                if pred is not None:
                    add = 1 if (pred[0] == y_cp) else 0
                    error = self.adwin_ensemble[i].estimation
                    self.adwin_ensemble[i].add_element(add)
                    if self.adwin_ensemble[i].detected_change():
                        if self.adwin_ensemble[i].estimation > error:
                            change_detected = True
            except ValueError:
                change_detected = False

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

    def __adjust_ensemble_size(self):
        if len(self.classes) != len(self.ensemble):
            if len(self.classes) > len(self.ensemble):
                for i in range(len(self.ensemble), len(self.classes)):
                    self.ensemble.append(cp.deepcopy(self.base_estimator))
                    self.adwin_ensemble.append(ADWIN(self.delta))
                    self.actual_n_estimators += 1

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
        """ Estimates the probability of each sample in X belonging to each of the class-labels.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The matrix of samples one wants to predict the class probabilities for.

        Raises
        ------
        ValueError: A ValueError is raised if the number of classes in the base
        learner exceed that of the ensemble learner.

        Returns
        -------
        A numpy.ndarray of shape (n_samples, n_labels), in which each outer entry is associated with the X entry of the
        same index. And where the list in index [i] contains len(self.target_values) elements, each of which represents
        the probability that the i-th sample of X belongs to a certain class-label.

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
                    raise ValueError("The number of classes in the base learner is larger than in the ensemble.")

                if len(proba) < 1:
                    for n in range(r):
                        proba.append([0.0] * len(partial_proba[n]))

                for n in range(r):
                    for l in range(len(partial_proba[n])):
                        try:
                            proba[n][l] += partial_proba[n][l]
                        except IndexError:
                            proba[n].append(partial_proba[n][l])
        except ValueError:
            return np.zeros((r, 1))
        except TypeError:
            return np.zeros((r, 1))

        # normalizing probabilities
        sum_proba = []
        for l in range(r):
            sum_proba.append(np.sum(proba[l]))
        aux = []
        for i in range(len(proba)):
            if sum_proba[i] > 0.:
                aux.append([x / sum_proba[i] for x in proba[i]])
            else:
                aux.append(proba[i])
        return np.asarray(aux)

    def predict_binary_proba(self, X):
        """ Calculates the probability of each sample belonging to each coded label.

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
        proba = []
        r, c = get_dimensions(X)
        if not self.init_matrix_codes:
            try:
                for i in range(self.actual_n_estimators):
                    vote = self.ensemble[i].predict_proba(X)
                    vote_class = 0

                    if len(vote) == 2:
                        vote_class = 1 if (vote[1] > vote[0]) else 0

                    if len(proba) < 1:
                        for n in range(r):
                            proba.append([0.0 for _ in vote[n]])

                    for j in range(len(self.classes)):
                        if self.matrix_codes[i][j] == vote_class:
                            proba[j] += 1
            except ValueError:
                return np.zeros((r, 1))

            if len(proba) < 1:
                return None

            # normalizing probabilities
            sum_proba = []
            for l in range(r):
                sum_proba.append(np.sum(proba[l]))
            aux = []
            for i in range(len(proba)):
                if sum_proba[i] > 0.:
                    aux.append([x / sum_proba[i] for x in proba[i]])
                else:
                    aux.append(proba[i])
            return aux
        return None

    def reset(self):
        """ Resets all the estimators, as well as all the ADWIN change detectors.
        
        Returns
        -------
        LeverageBagging
            self
        """
        self.__configure()
        return self
