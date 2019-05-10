import copy

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from skmultiflow.core import BaseSKMObject, ClassifierMixin, MetaEstimatorMixin
from skmultiflow.utils import check_random_state


class LearnPP(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):
    """ Learn++ ensemble classifier.

    Learn++ [1]_  does not require access to previously used data during
    subsequent incremental learning steps. At the same time, it does not
    forget previously acquired knowledge. Learn++ utilizes an ensemble of
    classifiers by generating multiple hypotheses using training data
    sampled according to carefully tailored distributions.

    References
    ----------
    .. [1] Polikar, Robi and Upda, Lalita and Upda, Satish S and Honavar, Vasant.
       Learn++: An Incremental Learning Algorithm for Supervised Neural Networks.
       IEEE Transactions on Systems Man and Cybernetics Part C (Applications and Reviews), 2002.

    Parameters
    ----------
    base_estimator: skmultiflow.core.BaseSKMObject or sklearn.BaseEstimator (default=DecisionTreeClassifier)
        Each member of the ensemble is an instance of the base estimator.
    n_estimators: int (default=30)
        The number of classifiers per ensemble
    n_ensembles: int (default=10)
        The number of ensembles to keep.
    window_size: int (default=100)
        The size of the training window (batch), in other words, how many instances are kept for training.
    error_threshold: float (default=0.5)
        Only keep the learner with the error smaller than error_threshold
    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by `np.random`.

    Raises
    ------
    RuntimeError:
        A RuntimeError is raised if the base_estimator is too weak. In other words,
        it has too low accuracy on the dataset. A RuntimeError is also raised if the
        'classes' parameter is not passed in the first partial_fit call, or if they
        are passed in further calls but differ from the initial classes.

    Examples
    --------
    >>> # Imports
    >>> import numpy as np
    >>> from skmultiflow.meta.learn_pp import LearnPP
    >>> from skmultiflow.lazy.knn import KNN
    >>> from skmultiflow.data.sea_generator import SEAGenerator
    >>> # Setting up the stream
    >>> stream = SEAGenerator(1)
    >>> stream.prepare_for_use()
    >>> # Setting up the Learn++ classifier to work with KNN classifiers
    >>> clf = LearnPP(base_estimator=KNN(n_neighbors=8, max_window_size=2000, leaf_size=30), n_estimators=30)
    >>> # Keeping track of sample count and correct prediction count
    >>> sample_count = 0
    >>> corrects = 0
    >>> m = 200
    >>> # Pre training the classifier with 200 samples
    >>> X, y = stream.next_sample(m)
    >>> clf = clf.partial_fit(X, y, classes=stream.target_values)
    >>> for i in range(3):
    ...     X, y = stream.next_sample(m)
    ...     pred = clf.predict(X)
    ...     clf = clf.partial_fit(X, y)
    ...     if pred is not None:
    ...         corrects += np.sum(y == pred)
    ...     sample_count += m
    >>>
    >>> # Displaying the results
    >>> print('Learn++ classifier performance: ' + str(corrects / sample_count))
    Learn++ classifier performance: 0.9555

    """

    def __init__(self, base_estimator=DecisionTreeClassifier(),
                 error_threshold=0.5,
                 n_estimators=30,
                 n_ensembles=10,
                 window_size=100,
                 random_state=None):
        super().__init__()
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.ensembles = []
        self.ensemble_weights = []
        self.classes = None
        self.n_ensembles = n_ensembles
        self.random = check_random_state(random_state)
        self.random_state = random_state
        self.error_threshold = error_threshold
        self.X_batch = []
        self.y_batch = []
        self.window_size = window_size

    def reset(self):
        self.ensembles = []
        self.ensemble_weights = []
        self.X_batch = []
        self.y_batch = []
        self.random = check_random_state(self.random_state)

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

        sample_weight: NOT used (default=None)

        Raises
        ------
        RuntimeError:
            A RuntimeError is raised if the 'classes' parameter is not
            passed in the first partial_fit call, or if they are passed in further
            calls but differ from the initial classes list passed.
            A RuntimeError is raised if the base_estimator is too weak. In other word,
            it has too low accuracy on the dataset.

        Returns
        -------
        LearnPP
            self
        """

        if self.classes is None:
            if classes is None:
                raise RuntimeError("Should pass the classes in the first partial_fit call")
            else:
                self.classes = classes

        if classes is not None and self.classes is not None:
            if set(classes) == set(self.classes):
                pass
            else:
                raise RuntimeError("The values of classes are different")

        N, _ = X.shape

        for i in range(N):
            self.X_batch.append(X[i])
            self.y_batch.append(y[i])
            if len(self.y_batch) == self.window_size:
                self.__fit_batch(self.X_batch, self.y_batch)
                self.X_batch = []
                self.y_batch = []

        return self

    def __fit_batch(self, X, y):
        ensemble = [copy.deepcopy(self.base_estimator) for _ in range(self.n_estimators)]
        normalized_errors = [1.0 for _ in range(self.n_estimators)]

        m = len(X)
        X = np.array(X)
        y = np.array(y)

        Dt = np.ones((m,)) / m

        items_index = np.linspace(0, m - 1, m)
        t = 0
        while t < self.n_estimators:
            patience = 0

            # Set distribution Dt
            Dt = Dt / np.sum(Dt)

            total_error = 1.0
            while total_error >= self.error_threshold:

                # create training and testing subsets according to Dt
                train_size = int(m / 2)
                test_size = int(m / 2)
                train_items_index = self._get_item(items_index, Dt, train_size)
                test_items_index = self._get_item(items_index, Dt, test_size)

                X_train = X[train_items_index]
                y_train = y[train_items_index]
                X_test = X[test_items_index]
                y_test = y[test_items_index]

                # Train a weak learner
                ensemble[t] = copy.deepcopy(self.base_estimator)
                try:
                    ensemble[t].fit(X_train, y_train)
                except NotImplementedError:
                    ensemble[t].partial_fit(X_train, y_train)

                # predict on the data
                y_predict = ensemble[t].predict(X_test)

                total_error = self.__compute_error(Dt[test_items_index], y_test, y_predict)

                if total_error < self.error_threshold:

                    norm_error = total_error / (1 - total_error)
                    normalized_errors[t] = norm_error

                    # predict using all hypothesis in the ensemble with majority votes
                    y_predict_composite = self.__majority_vote(X, t + 1, ensemble, normalized_errors)

                    total_error = self.__compute_error(Dt, y, y_predict_composite)
                    if total_error < self.error_threshold:
                        normalize_composite_error = total_error / (1 - total_error)
                        if t < self.n_estimators - 1:
                            Dt[y_predict_composite == y] = Dt[y_predict_composite == y] * normalize_composite_error

                if total_error > self.error_threshold:
                    patience += 1
                else:
                    patience = 0
                if patience > 1000:
                    raise RuntimeError("Your base estimator is too weak")
            t += 1

        self.ensembles.append(ensemble)
        self.ensemble_weights.append(normalized_errors)

        if len(self.ensembles) > self.n_ensembles:
            self.ensembles.pop(0)
            self.ensemble_weights.pop(0)

        return self

    @staticmethod
    def __compute_error(Dt, y_true, y_predict):
        total_error = np.sum(Dt[y_predict != y_true]) / np.sum(Dt)
        return total_error

    def __vote_proba(self, X, t, ensemble, normalized_errors):
        res = []
        for m in range(len(X)):
            votes = np.zeros(len(self.classes))
            for i in range(t):
                h = ensemble[i]

                y_predicts = h.predict(X[m].reshape(1, -1))
                norm_error = normalized_errors[i]
                votes[int(y_predicts[0])] += np.log(1 / (norm_error + 1e-50))

            res.append(votes)
        return res

    def __majority_vote(self, X, t, ensemble, normalized_errors):
        res = self.__vote_proba(X, t, ensemble, normalized_errors)
        return np.argmax(res, axis=1)

    def predict_proba(self, X):
        """ Predicts the probability of each sample belonging to each one of the
        known classes.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            A matrix of the samples we want to predict.

        Returns
        -------
        numpy.ndarray
            An array of shape (n_samples, n_features), in which each outer entry is
            associated with the X entry of the same index. And where the list in
            index [i] contains len(self.target_values) elements, each of which represents
            the probability that the i-th sample of X belongs to a certain label.

        """
        votes = np.zeros((len(X), len(self.classes)))
        for i in range(len(self.ensembles)):
            ensemble = self.ensembles[i]
            ensemble_weight = self.ensemble_weights[i]
            votes += np.array(self.__vote_proba(X, self.n_estimators, ensemble, ensemble_weight))
        return votes

    def predict(self, X):
        """ Predict classes for the passed data.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The set of data samples to predict the labels for.

        Returns
        -------
        A numpy.ndarray with all the predictions for the samples in X.

        Notes
        -----

        The predict function uses majority votes from all its learners
        with their weights to find the most likely prediction for the sample matrix X.

        """
        votes = self.predict_proba(X)
        return np.argmax(votes, axis=1)

    def _get_item(self, items, items_weights, number_of_items):
        return self.random.choice(items, number_of_items, p=items_weights).astype(np.int32)
