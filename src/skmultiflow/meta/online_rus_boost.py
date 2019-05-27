import copy as cp

from skmultiflow.core import BaseSKMObject, ClassifierMixin, MetaEstimatorMixin
from skmultiflow.drift_detection import ADWIN
from skmultiflow.lazy import KNNAdwin
from skmultiflow.utils import check_random_state
from skmultiflow.utils.utils import *


class OnlineRUSBoost(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):
    """ Online RUSBoost ensemble classifier.

    Online RUSBoost [1]_ is the adaptation of the ensemble learner to data streams.

    RUSBoost is a follows the techniques of UnderOverBagging and SMOTEBagging by
    introducing sampling techniques as a postprocessing step before each iteration of
    the standard AdaBoost algorithm. RUSBoost uses three implementations:

        - RUSBoost 1: fixes the class ration.
        - RUSBoosT 2: fixes the example distribution.
        - RUSBoost 3: fixes the sampling rate.

    This online ensemble learner method is improved by the addition of an ADWIN change
    detector.

    ADWIN stands for Adaptive Windowing. It works by keeping updated
    statistics of a variable sized window, so it can detect changes and
    perform cuts in its window to better adapt the learning algorithms.

    Parameters
    ----------
    base_estimator: skmultiflow.core.BaseSKMObject or sklearn.BaseEstimator (default=KNNAdwin)
        Each member of the ensemble is an instance of the base estimator.

    n_estimators: int, optional (default=10)
        The size of the ensemble, in other words, how many classifiers to train.

    sampling_rate: int, optional (default=3)
        The sampling rate of the positive instances.

    algorithm: int, optional (default=1)
        The implementation of RUSBoost to use {1, 2, 3}.

    drift_detection: bool, optional (default=True)
        A drift detector (ADWIN) can be used by the method to track the performance
         of the classifiers and adapt when a drift is detected.

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

    References
    ----------
    .. [1] B. Wang and J. Pineau, "Online Bagging and Boosting for Imbalanced Data Streams,"
       in IEEE Transactions on Knowledge and Data Engineering, vol. 28, no. 12, pp. 3353-3366,
       1 Dec. 2016. doi: 10.1109/TKDE.2016.2609424

    """

    def __init__(self,
                 base_estimator=KNNAdwin(),
                 n_estimators=10,
                 sampling_rate=3,
                 algorithm=1,
                 drift_detection=True,
                 random_state=None):

        super().__init__()
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.sampling_rate = sampling_rate
        self.algorithm = algorithm
        self.drift_detection = drift_detection
        # default values
        self.ensemble = None
        self.actual_n_estimators = None
        self.classes = None
        self._random_state = None
        self.adwin_ensemble = None
        self.lam_sc = None
        self.lam_pos = None
        self.lam_neg = None
        self.lam_sw = None
        self.epsilon = None
        self.__configure()

    def __configure(self):
        if hasattr(self.base_estimator, "reset"):
            self.base_estimator.reset()

        self.actual_n_estimators = self.n_estimators
        self.adwin_ensemble = []
        for i in range(self.actual_n_estimators):
            self.adwin_ensemble.append(ADWIN())
        self.ensemble = [cp.deepcopy(self.base_estimator) for _ in range(self.actual_n_estimators)]
        self._random_state = check_random_state(self.random_state)
        self.lam_sc = np.zeros(self.actual_n_estimators)
        self.lam_pos = np.zeros(self.actual_n_estimators)
        self.lam_neg = np.zeros(self.actual_n_estimators)
        self.lam_sw = np.zeros(self.actual_n_estimators)
        self.epsilon = np.zeros(self.actual_n_estimators)
        self.n_pos = 0
        self.n_neg = 0

    def reset(self):
        self.__configure()

    def partial_fit(self, X, y, classes=None, weight=None):
        """ Partially fits the model, based on the X and y matrix.

        Since it's an ensemble learner, if X and y matrix of more than one
        sample are passed, the algorithm will partial fit the model one sample
        at a time.

        Each sample is trained by each classifier a total of K times, where K
        is drawn by a Poisson(l) distribution. l is updated after every example
        using :math:`lambda_{sc}` if th estimator correctly classifies the example or
        :math:`lambda_{sw}` in the other case.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.

        y: numpy.ndarray of shape (n_samples)
            An array-like with the class labels of all samples in X.

        classes: numpy.ndarray, optional (default=None)
            Array with all possible/known class labels. This is an optional parameter, except
            for the first partial_fit call where it is compulsory.

        sample_weight: Array-like
            Instance weight. If not provided, uniform weights are assumed. Usage varies depending on the base estimator.

        Raises
        ------
        ValueError: A ValueError is raised if the 'classes' parameter is not
        passed in the first partial_fit call, or if they are passed in further
        calls but differ from the initial classes list passed.

        Returns
        -------
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
        r, _ = get_dimensions(X)
        for j in range(r):
            change_detected = False
            lam = 1
            for i in range(self.actual_n_estimators):
                if y[j] == 1:
                    self.lam_pos[i] += lam
                    self.n_pos += 1
                else:
                    self.lam_neg[i] += lam
                    self.n_neg += 1
                lam_rus = 1
                if self.algorithm == 1:
                    if y[j] == 1:
                        if self.n_neg != 0:
                            lam_rus = lam * ((self.lam_pos[i] + self.lam_neg[i]) /
                                             (self.lam_pos[i] + self.lam_neg[i] *
                                              (self.sampling_rate * (self.n_pos / self.n_neg))) *
                                             (((self.sampling_rate + 1) * self.n_pos) / (self.n_pos + self.n_neg)))
                    else:
                        if self.n_pos != 0:
                            lam_rus = lam * ((self.lam_pos[i] + self.lam_neg[i]) /
                                             (self.lam_pos[i] + self.lam_neg[i] *
                                              (self.n_neg / (self.n_pos * self.sampling_rate))) *
                                             (((self.sampling_rate + 1) * self.n_pos) / (self.n_pos + self.n_neg)))
                elif self.algorithm == 2:
                    if y[j] == 1:
                        lam_rus = ((lam * self.n_pos) / (self.n_pos + self.n_neg)) / \
                                  (self.lam_pos[i] / (self.lam_pos[i] + self.lam_neg[i]))
                    else:
                        lam_rus = ((lam * self.sampling_rate * self.n_pos) / (self.n_pos + self.n_neg)) / \
                                  (self.lam_neg[i] / (self.lam_pos[i] + self.lam_neg[i]))
                elif self.algorithm == 3:
                    if y[j] == 1:
                        lam_rus = lam
                    else:
                        lam_rus = lam / self.sampling_rate
                k = self._random_state.poisson(lam_rus)
                if k > 0:
                    for b in range(k):
                        self.ensemble[i].partial_fit([X[j]], [y[j]], classes, weight)
                if self.ensemble[i].predict([X[j]])[0] == y[j]:
                    self.lam_sc[i] += lam
                    self.epsilon[i] = (self.lam_sw[i]) / (self.lam_sc[i] + self.lam_sw[i])
                    if self.epsilon[i] != 1:
                        lam = lam / (2 * (1 - self.epsilon[i]))
                else:
                    self.lam_sw[i] += lam
                    self.epsilon[i] = (self.lam_sw[i]) / (self.lam_sc[i] + self.lam_sw[i])
                    if self.epsilon[i] != 0:
                        lam = lam / (2 * self.epsilon[i])

                if self.drift_detection:
                    try:
                        pred = self.ensemble[i].predict(X)
                        error_estimation = self.adwin_ensemble[i].estimation
                        for k in range(r):
                            if pred[k] is not None:
                                self.adwin_ensemble[i].add_element(int(pred[k] == y[k]))
                        if self.adwin_ensemble[i].detected_change():
                            if self.adwin_ensemble[i].estimation > error_estimation:
                                change_detected = True
                    except ValueError:
                        change_detected = False
                        pass

            if change_detected and self.drift_detection:
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
                    self.actual_n_estimators += 1
                    self.adwin_ensemble.append(ADWIN())
                self.lam_sc = np.zeros(self.actual_n_estimators)
                self.lam_pos = np.zeros(self.actual_n_estimators)
                self.lam_neg = np.zeros(self.actual_n_estimators)
                self.lam_sw = np.zeros(self.actual_n_estimators)
                self.epsilon = np.zeros(self.actual_n_estimators)

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
            for i in range(self.actual_n_estimators):
                partial_proba = self.ensemble[i].predict_proba(X)
                if len(partial_proba[0]) > max(self.classes) + 1:
                    raise ValueError("The number of classes in the base learner is larger than in the ensemble.")

                if len(proba) < 1:
                    for n in range(r):
                        proba.append([0.0 for _ in partial_proba[n]])

                for n in range(r):
                    for l in range(len(partial_proba[n])):
                        try:
                            proba[n][l] += np.log((1 - self.epsilon[i]) / self.epsilon[i]) * partial_proba[n][l]
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
