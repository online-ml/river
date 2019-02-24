import copy as cp
from skmultiflow.core.base import StreamModel
from skmultiflow.lazy.knn import KNN
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.utils.utils import *
from skmultiflow.utils import check_random_state


class LeverageBagging(StreamModel):
    """ Leverage Bagging Classifier
    
    An ensemble method, which represents an improvement from the online Oza 
    Bagging algorithm. The complete description of this method can be found 
    in Bifet, Holmes, and Pfahringer's 'Leveraging Bagging for Evolving Data 
    Streams'.
    
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
    
    Parameters
    ----------
    base_estimator: StreamModel
        This is the ensemble classifier type, each ensemble classifier will be a
        copy of the base_estimator.
        
    n_estimators: int
        The size of the ensemble, in other words, how many classifiers to train.
        
    w: int
        The poisson distribution's parameter, which is used to simulate 
        re-sampling.
        
    delta: float
        The delta parameter for the ADWIN change detector.
    
    enable_code_matrix: bool
        If set to True it will enable the output detection code matrix.
    
    leverage_algorithm: string 
        A string representing the bagging algorithm to use. Can be one of the 
        following: 'leveraging_bag', 'leveraging_bag_me', 'leveraging_bag_half', 
        'leveraging_bag_wt', 'leveraging_subag'

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by `np.random`.
    
    Raises
    ------
    NotImplementedError: A few of the functions described here are not 
    implemented since they have no application in this context.
    
    ValueError: A ValueError is raised if the 'classes' parameter is
    not passed in the first partial_fit call.

    
    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.meta import LeverageBagging
    >>> from skmultiflow.lazy.knn import KNN
    >>> from skmultiflow.data.sea_generator import SEAGenerator
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
        self.random_state = None
        self.base_estimator = base_estimator
        self._init_n_estimators = n_estimators
        self.enable_matrix_codes = enable_code_matrix
        self.w = w
        self.delta = delta
        if leverage_algorithm not in self.LEVERAGE_ALGORITHMS:
            raise ValueError("Leverage algorithm not supported.")
        self.leveraging_algorithm = leverage_algorithm
        self._init_random_state = random_state
        self.__configure()

    def __configure(self):
        self.base_estimator.reset()
        self.n_estimators = self._init_n_estimators
        self.ensemble = [cp.deepcopy(self.base_estimator) for _ in range(self.n_estimators)]
        self.adwin_ensemble = []
        for i in range(self.n_estimators):
            self.adwin_ensemble.append(ADWIN(self.delta))
        self.random_state = check_random_state(self._init_random_state)
        self.n_detected_changes = 0
        self.classes = None
        self.init_matrix_codes = True

    def fit(self, X, y, classes=None, weight=None):
        raise NotImplementedError

    def partial_fit(self, X, y, classes=None, weight=None):
        """ partial_fit
        
        Partially fit the ensemble's method models. 
        
        This id done by calling the private funcion __partial_fit.
        
        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            The samples used to update the models.
            
        y: Array-like
            An array containing all the labels for the samples in X.
            
        classes: list
            A list with all the possible labels of the classification task.
            It's an optional parameter, except for the first partial_fit 
            call, when it's a requirement.

        weight: Not used.
        
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
            if self.classes == set(classes):
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
            self.matrix_codes = np.zeros((self.n_estimators, len(self.classes)), dtype=int)
            for i in range(self.n_estimators):
                n_zeros = 0
                n_ones = 0
                while (n_ones - n_zeros) * (n_ones - n_zeros) > self.n_estimators % 2:
                    n_zeros = 0
                    n_ones = 0
                    for j in range(len(self.classes)):
                        if (j == 1) and (len(self.classes) == 2):
                            result = 1 - self.matrix_codes[i][0]
                        else:
                            result = self.random_state.randint(2)

                        self.matrix_codes[i][j] = result
                        if result == 1:
                            n_ones += 1
                        else:
                            n_zeros += 1
            self.init_matrix_codes = False

        change_detected = False
        X_cp, y_cp = cp.deepcopy(X), cp.deepcopy(y)
        for i in range(self.n_estimators):
            k = 0.0

            if self.leveraging_algorithm == self.LEVERAGE_ALGORITHMS[0]:
                k = self.random_state.poisson(self.w)

            elif self.leveraging_algorithm == self.LEVERAGE_ALGORITHMS[1]:
                error = self.adwin_ensemble[i].estimation
                pred = self.ensemble[i].predict(np.asarray([X]))
                if pred is None:
                    k = 1.0
                elif pred[0] != y:
                    k = 1.0
                elif self.random_state.rand() < (error/(1.0 - error)):
                    k = 1.0
                else:
                    k = 0.0

            elif self.leveraging_algorithm == self.LEVERAGE_ALGORITHMS[2]:
                w = 1.0
                k = 0.0 if (self.random_state.randint(2) == 1) else w

            elif self.leveraging_algorithm == self.LEVERAGE_ALGORITHMS[3]:
                w = 1.0
                k = 1.0 + self.random_state.poisson(w)

            elif self.leveraging_algorithm == self.LEVERAGE_ALGORITHMS[4]:
                w = 1.0
                k = self.random_state.poisson(1)
                k = w if k > 0 else 0

            if k > 0:
                if self.enable_matrix_codes:
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
            for i in range(self.n_estimators):
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
                    self.n_estimators += 1

    def predict(self, X):
        """ predict
        
        Predicts the labels from all samples in the X matrix.
        
        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            A matrix of the samples we want to predict.
        
        Returns
        -------
        numpy.ndarray
            A numpy.ndarray with the label prediction for all the samples in X.
        
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
        """ predict_proba

        Calculates the probability of each sample in X belonging to each 
        of the labels, based on the knn algorithm. This is done by predicting 
        the class probability for each one of the ensemble's classifier, and 
        then taking the absolute probability from the ensemble itself.

        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            All the samples we want to predict the label for.

        Raises
        ------
        ValueError: A ValueError is raised if the number of classes in the base
        learner exceed that of the ensemble learner.

        Returns
        -------
        numpy.ndarray
            An array of shape (n_samples, n_features), in which each outer entry is 
            associated with the X entry of the same index. And where the list in 
            index [i] contains len(self.classes) elements, each of which represents
            the probability that the i-th sample of X belongs to a certain label.

        """
        if self.enable_matrix_codes:
            return self.predict_binary_proba(X)
        proba = []
        r, c = get_dimensions(X)
        try:
            for i in range(self.n_estimators):
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
        """ predict_binary_proba

        Calculates the probability of each sample in X belonging to each 
        coded label. This will only be used if matrix codes are enabled. 
        Otherwise the method will use the normal predict_proba function.

        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
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
                for i in range(self.n_estimators):
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

    def score(self, X, y):
        raise NotImplementedError

    def get_info(self):
        return 'LeverageBagging Classifier: base_estimator: ' + str(type(self.base_estimator).__name__) + \
               ' - n_estimators: ' + str(self.n_estimators) + \
               ' - w: ' + str(self.w) + \
               ' - delta: ' + str(self.delta) + \
               ' - enable_code_matrix: ' + ('True' if self.enable_matrix_codes else 'False') + \
               ' - leveraging_algorithm: ' + self.leveraging_algorithm
