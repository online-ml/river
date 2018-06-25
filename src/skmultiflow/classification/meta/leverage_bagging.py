import copy as cp
from skmultiflow.core.base import StreamModel
from skmultiflow.classification.lazy.knn import KNN
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.core.utils.utils import *


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
    h: classifier (extension of the BaseClassifier)
        This is the ensemble classifier type, each ensemble classifier is going 
        to be a copy of the h classifier.
        
    ensemble_length: int
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
    
    Raises
    ------
    NotImplementedError: A few of the functions described here are not 
    implemented since they have no application in this context.
    
    ValueError: A ValueError is raised if the 'classes' parameter is
    not passed in the first partial_fit call.
    
    Notes
    -----
    To choose the correct ensemble_length (a value too high or too low may 
    deteriorate performance) there are different techniques. One of them is 
    called 'The law of diminishing returns in ensemble construction' by Bonab 
    and Can. This theoretical framework claims, with experimental results, that 
    the optimal number of classifiers in an online ensemble method is equal to 
    the number of labels in the classification task. Thus we chose a default 
    value of 2, adapted to binary classification tasks.
    
    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.classification.meta.leverage_bagging import LeverageBagging
    >>> from skmultiflow.classification.lazy.knn import KNN
    >>> from skmultiflow.data.generators.sea_generator import SEAGenerator
    >>> # Setting up the stream
    >>> stream = SEAGenerator(1, noise_percentage=6.7)
    >>> stream.prepare_for_use()
    >>> # Setting up the LeverageBagging classifier to work with KNN classifiers
    >>> clf = LeverageBagging(h=KNN(k=8, max_window_size=2000, leaf_size=30), ensemble_length=2)
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
    def __init__(self, h=KNN(), ensemble_length=2, w=6, delta=0.002, enable_code_matrix=False,
                 leverage_algorithm='leveraging_bag'):

        super().__init__()
        # default values
        self.h = h.reset()
        self.ensemble_length = None
        self.ensemble = None
        self.adwin_ensemble = None
        self.n_detected_changes = None
        self.matrix_codes = None
        self.enable_matrix_codes = None
        self.w = None
        self.delta = None
        self.classes = None
        self.leveraging_algorithm = None
        self.__configure(h, ensemble_length, w, delta, enable_code_matrix, leverage_algorithm)
        self.init_matrix_codes = True

        self.adwin_ensemble = []
        for i in range(ensemble_length):
            self.adwin_ensemble.append(ADWIN(self.delta))

    def __configure(self, h, ensemble_length, w, delta, enable_code_matrix, leverage_algorithm):
        self.ensemble_length = ensemble_length
        self.ensemble = [cp.deepcopy(h) for x in range(ensemble_length)]
        self.w = w
        self.delta = delta
        self.enable_matrix_codes = enable_code_matrix
        if leverage_algorithm not in self.LEVERAGE_ALGORITHMS:
            raise ValueError("Leverage algorithm not supported.")
        self.leveraging_algorithm = leverage_algorithm

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
        n_classes = len(self.classes)
        change = False

        if self.init_matrix_codes:
            self.matrix_codes = np.zeros((self.ensemble_length, len(self.classes)), dtype=int)
            for i in range(self.ensemble_length):
                n_zeros = 0
                n_ones = 0
                while((n_ones - n_zeros) * (n_ones - n_zeros) > self.ensemble_length % 2):
                    n_zeros = 0
                    n_ones = 0
                    for j in range(len(self.classes)):
                        result = 0
                        if (j == 1) and (len(self.classes) == 2):
                            result = 1 - self.matrix_codes[i][0]
                        else:
                            result = np.random.randint(2)

                        self.matrix_codes[i][j] = result
                        if result == 1:
                            n_ones += 1
                        else:
                            n_zeros += 1
            self.init_matrix_codes = False

        detected_change = False
        X_cp, y_cp = cp.deepcopy(X), cp.deepcopy(y)
        for i in range(self.ensemble_length):
            k = 0.0

            if self.leveraging_algorithm == self.LEVERAGE_ALGORITHMS[0]:
                k = np.random.poisson(self.w)

            elif self.leveraging_algorithm == self.LEVERAGE_ALGORITHMS[1]:
                error = self.adwin_ensemble[i]._estimation
                pred = self.ensemble[i].predict(np.asarray([X]))
                if pred is None:
                    k = 1.0
                elif pred[0] != y:
                    k = 1.0
                elif np.random.rand() < (error/(1.0 - error)):
                    k = 1.0
                else:
                    k = 0.0

            elif self.leveraging_algorithm == self.LEVERAGE_ALGORITHMS[2]:
                w = 1.0
                k = 0.0 if (np.random.randint(2) == 1) else w

            elif self.leveraging_algorithm == self.LEVERAGE_ALGORITHMS[3]:
                w = 1.0
                k = 1.0 + np.random.poisson(w)

            elif self.leveraging_algorithm == self.LEVERAGE_ALGORITHMS[4]:
                w = 1.0
                k = np.random.poisson(1)
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
                    error = self.adwin_ensemble[i]._estimation
                    self.adwin_ensemble[i].add_element(add)
                    if self.adwin_ensemble[i].detected_change():
                        if self.adwin_ensemble[i]._estimation > error:
                            change = True
            except ValueError:
                change = False

        if change:
            self.n_detected_changes += 1
            max = 0.0
            imax = -1
            for i in range(self.ensemble_length):
                if max < self.adwin_ensemble[i]._estimation:
                    max = self.adwin_ensemble[i]._estimation
                    imax = i
            if imax != -1:
                self.ensemble[imax].reset()
                self.adwin_ensemble[imax] = ADWIN(self.delta)
        return self

    def __adjust_ensemble_size(self):
        if len(self.classes) != len(self.ensemble):
            if len(self.classes) > len(self.ensemble):
                for i in range(len(self.ensemble), len(self.classes)):
                    self.ensemble.append(cp.deepcopy(self.h))
                    self.adwin_ensemble.append(ADWIN(self.delta))
                    self.ensemble_length += 1

    def predict(self, X):
        """ predict
        
        Predicts the labels from all samples in the X matrix.
        
        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            A matrix of the samples we want to predict.
        
        Returns
        -------
        list
            A list with the label prediction for all the samples in X.
        
        """
        r, c = get_dimensions(X)
        probs = self.predict_proba(X)
        preds = []
        if probs is None:
            return None
        for i in range(r):
            preds.append(self.classes[probs[i].index(np.max(probs[i]))])
        return preds

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
        ValueError: A ValueError is raised if the number of classes in the h
        learner differs from that of the ensemble learner.

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
        probs = []
        r, c = get_dimensions(X)
        try:
            for i in range(self.ensemble_length):
                partial_probs = self.ensemble[i].predict_proba(X)
                if len(partial_probs[0]) != len(self.classes):
                    raise ValueError(
                        "The number of classes is different in the bagging algorithm and in the chosen learning "
                        "algorithm.")

                if len(probs) < 1:
                    for n in range(r):
                        probs.append([0.0 for x in partial_probs[n]])

                for n in range(r):
                    for l in range(len(partial_probs[n])):
                        probs[n][l] += partial_probs[n][l]
        except ValueError:
            return None

        # normalizing probabilities
        total_sum = np.sum(probs)
        aux = []
        for i in range(len(probs)):
            aux.append([x / total_sum for x in probs[i]])
        return aux

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
        probs = []
        r, c = get_dimensions(X)
        if not self.init_matrix_codes:
            try:
                for i in range(self.ensemble_length):
                    vote = self.ensemble[i].predict_proba(X)
                    vote_class = 0

                    if len(vote) == 2:
                        vote_class = 1 if (vote[1] > vote[0]) else 0

                    if len(probs) < 1:
                        for n in range(r):
                            probs.append([0.0 for x in vote[n]])

                    for j in range(len(self.classes)):
                        if self.matrix_codes[i][j] == vote_class:
                            probs[j] += 1
            except ValueError:
                return None

            if len(probs) < 1:
                return None

            # normalizing probabilities
            if r > 1:
                total_sum = []
                for l in range(r):
                    total_sum.append(np.sum(probs[l]))
            else:
                total_sum = [np.sum(probs)]
            aux = []
            for i in range(len(probs)):
                aux.append([x / total_sum[i] for x in probs[i]])
            return aux
        return None

    def reset(self):
        """ reset
        
        Resets all the classifiers, as well as all the ADWIN change
        detectors.
        
        Returns
        -------
        LeverageBagging
            self
        
        """
        self.__configure(self.h, self.ensemble_length, self.w, self.delta, self.enable_matrix_codes)
        self.adwin_ensemble = []
        for i in range(self.ensemble_length):
            self.adwin_ensemble.append(ADWIN(self.delta))
        self.n_detected_changes = 0
        self.classes = None
        self.init_matrix_codes = True

        return self

    def score(self, X, y):
        raise NotImplementedError

    def get_info(self):
        return 'LeverageBagging Classifier: h: ' + str(self.h) + \
               ' - ensemble_length: ' + str(self.ensemble_length) + \
               ' - w: ' + str(self.w) + \
               ' - delta: ' + str(self.delta) + \
               ' - enable_code_matrix: ' + ('True' if self.enable_matrix_codes else 'False') + \
               ' - leveraging_algorithm: ' + self.leveraging_algorithm
