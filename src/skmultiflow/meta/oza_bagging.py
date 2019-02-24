import copy as cp
from skmultiflow.core.base import StreamModel
from skmultiflow.lazy import KNNAdwin
from skmultiflow.utils.utils import *
from skmultiflow.utils import check_random_state


class OzaBagging(StreamModel):
    """ OzaBagging Classifier
    
    Oza Bagging is an ensemble learning method first introduced by Oza and 
    Russel's 'Online Bagging and Boosting'. They are an improvement of the 
    well known Bagging ensemble method for the batch setting, which in this 
    version can effectively handle data streams.
    
    For a traditional Bagging algorithm, adapted for the batch setting, we 
    would have M classifiers training on M different datasets, created by 
    drawing N samples from the N-sized training set with replacement.
    
    In the online context, since there is no training dataset, but a stream 
    of samples, the drawing of samples with replacement can't be trivially 
    executed. The strategy adopted by the Online Bagging algorithm is to 
    simulate this task by training each arriving sample K times, which is 
    drawn by the binomial distribution. Since we can consider the data stream 
    to be infinite, and knowing that with infinite samples the binomial 
    distribution tends to a Poisson(1) distribution, Oza and Russel found 
    that to be a good 'drawing with replacement'.
    
    Parameters
    ----------
    base_estimator: StreamModel
        This is the ensemble classifier type, each ensemble classifier is going 
        to be a copy of the base_estimator.
    
    n_estimators: int
        The size of the ensemble, in other words, how many classifiers to train.

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
    >>> from skmultiflow.meta.oza_bagging import OzaBagging
    >>> from skmultiflow.lazy.knn import KNN
    >>> from skmultiflow.data.sea_generator import SEAGenerator
    >>> # Setting up the stream
    >>> stream = SEAGenerator(1, noise_percentage=6.7)
    >>> stream.prepare_for_use()
    >>> # Setting up the OzaBagging classifier to work with KNN classifiers
    >>> clf = OzaBagging(base_estimator=KNN(n_neighbors=8, max_window_size=2000, leaf_size=30), n_estimators=2)
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
    >>> print('OzaBagging classifier performance: ' + str(corrects / sample_count))
    OzaBagging classifier performance: 0.9645
    
    """

    def __init__(self, base_estimator=KNNAdwin(), n_estimators=10, random_state=None):
        super().__init__()
        # default values
        self.ensemble = None
        self.n_estimators = None
        self.classes = None
        self.random_state = None
        self._init_n_estimators = n_estimators
        self._init_random_state = random_state
        self.__configure(base_estimator)

    def __configure(self, base_estimator):
        base_estimator.reset()
        self.base_estimator = base_estimator
        self.n_estimators = self._init_n_estimators
        self.ensemble = [cp.deepcopy(base_estimator) for _ in range(self.n_estimators)]
        self.random_state = check_random_state(self._init_random_state)

    def reset(self):
        self.__configure(self.base_estimator)

    def fit(self, X, y, classes=None, weight=None):
        raise NotImplementedError

    def partial_fit(self, X, y, classes=None, weight=None):
        """ partial_fit
         
        Partially fits the model, based on the X and y matrix.
                
        Since it's an ensemble learner, if X and y matrix of more than one 
        sample are passed, the algorithm will partial fit the model one sample 
        at a time.
        
        Each sample is trained by each classifier a total of K times, where K 
        is drawn by a Poisson(1) distribution.
        
        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features) 
            Features matrix used for partially updating the model.
            
        y: Array-like
            An array-like of all the class labels for the samples in X.
            
        classes: list 
            List of all existing classes. This is an optional parameter, except
            for the first partial_fit call, when it becomes obligatory.

        weight: Array-like
            Instance weight. If not provided, uniform weights are assumed.

        Raises
        ------
        ValueError: A ValueError is raised if the 'classes' parameter is not
        passed in the first partial_fit call, or if they are passed in further 
        calls but differ from the initial classes list passed.
        
        Returns
        _______
        OzaBagging
            self
        
        """
        if self.classes is None:
            if classes is None:
                raise ValueError("The first partial_fit call should pass all the classes.")
            else:
                self.classes = classes

        if self.classes is not None and classes is not None:
            if set(self.classes) == set(classes):
                pass
            else:
                raise ValueError("The classes passed to the partial_fit function differ from those passed earlier.")

        self.__adjust_ensemble_size()

        for i in range(self.n_estimators):
            k = self.random_state.poisson()
            if k > 0:
                for b in range(k):
                    self.ensemble[i].partial_fit(X, y, classes, weight)
        return self

    def __adjust_ensemble_size(self):
        if len(self.classes) != len(self.ensemble):
            if len(self.classes) > len(self.ensemble):
                for i in range(len(self.ensemble), len(self.classes)):
                    self.ensemble.append(cp.deepcopy(self.base_estimator))
                    self.n_estimators += 1

    def predict(self, X):
        """ predict
        
        The predict function will average the predictions from all its learners 
        to find the most likely prediction for the sample matrix X.
        
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
            return None
        for i in range(r):
            predictions.append(np.argmax(proba[i]))
        return np.asarray(predictions)

    def predict_proba(self, X):
        """ predict_proba
        
        Predicts the probability of each sample belonging to each one of the 
        known classes.
        
        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            A matrix of the samples we want to predict.
        
        Raises
        ------
        ValueError: A ValueError is raised if the number of classes in the base_estimator
        learner differs from that of the ensemble learner.
        
        Returns
        -------
        numpy.ndarray
            An array of shape (n_samples, n_features), in which each outer entry is 
            associated with the X entry of the same index. And where the list in 
            index [i] contains len(self.target_values) elements, each of which represents
            the probability that the i-th sample of X belongs to a certain label.
        
        """
        proba = []
        r, c = get_dimensions(X)
        try:
            for i in range(self.n_estimators):
                partial_proba = self.ensemble[i].predict_proba(X)
                if len(partial_proba[0]) > max(self.classes) + 1:
                    raise ValueError("The number of classes in the base learner is larger than in the ensemble.")

                if len(proba) < 1:
                    for n in range(r):
                        proba.append([0.0 for _ in partial_proba[n]])

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

    def score(self, X, y):
        raise NotImplementedError

    def get_info(self):
        return 'OzaBagging Classifier: base_estimator: ' + str(self.base_estimator) + \
               ' - n_estimators: ' + str(self.n_estimators)
