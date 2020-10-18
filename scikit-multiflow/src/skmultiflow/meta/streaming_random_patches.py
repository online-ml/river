from copy import deepcopy
from typing import List, Optional
from collections import deque

import numpy as np

from sklearn.preprocessing import normalize

from skmultiflow.core import BaseSKMObject, ClassifierMixin, MetaEstimatorMixin, clone
from river.base import DriftDetector
from skmultiflow.trees import HoeffdingTreeClassifier
from river.drift import ADWIN
from skmultiflow.utils import check_random_state, get_dimensions
from river.metrics import _ClassificationReport


class StreamingRandomPatchesClassifier(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):
    """ Streaming Random Patches ensemble classifier.

    Parameters
    ----------
    base_estimator: BaseSKMObject or sklearn.BaseObject, \
        (default=HoeffdingTreeClassifier)
        The base estimator.

    n_estimators: int, (default=100)
        Number of members in the ensemble.

    subspace_mode: str, (default='percentage')
        | Indicates how ``m``, defined by subspace_size, is interpreted.
          ``M``  represents the total number of features.
        | Only applies when training method is random subspaces or
          random patches.
        | 'm' - Specified value
        | 'sqrtM1' - ``sqrt(M)+1``
        | 'MsqrtM1' - ``M-(sqrt(M)+1)``
        | 'percentage' - Percentage

    subspace_size: int, (default=60)
        Number of features per subset for each classifier.
        Negative value means ``total_features - subspace_size``.

    training_method: str, (default='randompatches')
        | The training method to use.
        | 'randomsubspaces' - Random subspaces
        | 'resampling' - Resampling (bagging)
        | 'randompatches' - Random patches

    lam: float, (default=6.0)
        Lambda value for bagging.

    drift_detection_method: DriftDetector, (default=ADWIN(delta=1e-5))
        Drift detection method.

    warning_detection_method: DriftDetector, (default=ADWIN(delta=1e-4))
        Warning detection method.

    disable_weighted_vote: bool (default=False)
        If True, disables weighted voting.

    disable_drift_detection: bool (default=False)
        If True, disables drift detection and background learner.

    disable_background_learner: bool (default=False)
        If True, disables background learner and trees are reset
        immediately if drift is detected.

    nominal_attributes: list, optional
        List of Nominal attributes. If empty, then assume that all
        attributes are numerical.

    random_state: int, RandomState instance or None, optional (default=None)
       If int, random_state is the seed used by the random number generator;
       If RandomState instance, random_state is the random number generator;
       If None, the random number generator is the RandomState instance used
       by `np.random`.

    Notes
    -----
    The Streaming Random Patches (SRP) [1]_ ensemble method simulates bagging
    or random subspaces. The default algorithm uses both bagging and random
    subspaces, namely Random Patches. The default base estimator is a
    Hoeffding Tree, but it can be used with any other base estimator
    (differently from random forest variations).

    References
    ----------
    .. [1] Heitor Murilo Gomes, Jesse Read, Albert Bifet.
       Streaming Random Patches for Evolving Data Stream Classification.
       IEEE International Conference on Data Mining (ICDM), 2019.

    Examples
    --------
    >>> from skmultiflow.data import AGRAWALGenerator
    >>> from skmultiflow.meta import StreamingRandomPatchesClassifier
    >>>
    >>> stream = AGRAWALGenerator(random_state=1)
    >>> srp = StreamingRandomPatchesClassifier(random_state=1,
    >>>                                              n_estimators=3)
    >>>
    >>> # Variables to control loop and track performance
    >>> n_samples = 0
    >>> correct_cnt = 0
    >>> max_samples = 200
    >>>
    >>> # Run test-then-train loop for max_samples
    >>> # or while there is data in the stream
    >>> while n_samples < max_samples and stream.has_more_samples():
    >>>     X, y = stream.next_sample()
    >>>     y_pred = srp.predict(X)
    >>>     if y[0] == y_pred[0]:
    >>>         correct_cnt += 1
    >>>     srp.partial_fit(X, y)
    >>>     n_samples += 1
    >>>
    >>> print('{} samples analyzed.'.format(n_samples))

    """

    _TRAIN_RANDOM_SUBSPACES = "randomsubspaces"
    _TRAIN_RESAMPLING = "resampling"
    _TRAIN_RANDOM_PATCHES = "randompatches"

    _FEATURES_M = 'm'
    _FEATURES_SQRT = "sqrtM1"
    _FEATURES_SQRT_INV = "MsqrtM1"
    _FEATURES_PERCENT = "percentage"

    def __init__(self, base_estimator=HoeffdingTreeClassifier(grace_period=50,
                                                              split_confidence=0.01),
                 n_estimators: int = 100,
                 subspace_mode: str = "percentage",
                 subspace_size: int = 60,
                 training_method: str = "randompatches",
                 lam: float = 6.0,
                 drift_detection_method: DriftDetector = ADWIN(delta=1e-5),
                 warning_detection_method: DriftDetector = ADWIN(delta=1e-4),
                 disable_weighted_vote: bool = False,
                 disable_drift_detection: bool = False,
                 disable_background_learner: bool = False,
                 nominal_attributes=None,
                 random_state=None):

        self.base_estimator = base_estimator   # Not restricted to a specific base estimator.
        self.n_estimators = n_estimators
        if subspace_mode not in {self._FEATURES_SQRT, self._FEATURES_SQRT_INV,
                                 self._FEATURES_PERCENT, self._FEATURES_M}:
            raise ValueError("Invalid subspace_mode: {}.\n"
                             "Valid options are: {}".format(subspace_mode,
                                                            {self._FEATURES_M, self._FEATURES_SQRT,
                                                             self._FEATURES_SQRT_INV,
                                                             self._FEATURES_PERCENT}))
        self.subspace_mode = subspace_mode
        self.subspace_size = subspace_size
        if training_method not in {self._TRAIN_RESAMPLING, self._TRAIN_RANDOM_PATCHES,
                                   self._TRAIN_RANDOM_SUBSPACES}:
            raise ValueError("Invalid training_method: {}.\n"
                             "Valid options are: {}".format(training_method,
                                                            {self._TRAIN_RANDOM_PATCHES,
                                                             self._TRAIN_RANDOM_SUBSPACES,
                                                             self._TRAIN_RESAMPLING}))
        self.training_method = training_method
        self.lam = lam
        self.drift_detection_method = drift_detection_method
        self.warning_detection_method = warning_detection_method
        self.disable_weighted_vote = disable_weighted_vote
        self.disable_drift_detection = disable_drift_detection
        self.disable_background_learner = disable_background_learner
        # Single option (accuracy) for drift detection criteria. Could be extended in the future.
        self.drift_detection_criteria = 'accuracy'
        self.nominal_attributes = nominal_attributes if nominal_attributes else []
        self.random_state = random_state
        # self._random_state is the actual object used internally
        self._random_state = check_random_state(self.random_state)
        self.ensemble = None

        self._n_samples_seen = 0
        self._subspaces = None

        self._base_performance_evaluator = _ClassificationReport()
        self._base_learner_class = StreamingRandomPatchesBaseLearner

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """ Partially (incrementally) fit the model.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.

        y: numpy.ndarray of shape (n_samples)
            An array-like with the class labels of all samples in X.

        classes: numpy.ndarray, optional (default=None)
            No used.

        sample_weight: numpy.ndarray of shape (n_samples), optional \
            (default=None)
            Samples weight. If not provided, uniform weights are assumed.
            Usage varies depending on the learning method.

        Returns
        -------
        self

        """
        n_rows, n_cols = get_dimensions(X)

        if sample_weight is None:
            sample_weight = np.ones(n_rows)

        for i in range(n_rows):
            self._partial_fit(np.asarray([X[i]]), np.asarray([y[i]]),
                              classes=classes, sample_weight=np.asarray([sample_weight[i]]))

        return self

    def _partial_fit(self, X, y, classes=None, sample_weight=None):
        self._n_samples_seen += 1
        _, n_features = get_dimensions(X)

        if not self.ensemble:
            self._init_ensemble(n_features)

        for i in range(len(self.ensemble)):
            # Get prediction for instance
            y_pred = np.asarray([np.argmax(self.ensemble[i].predict_proba(X))])

            # Update performance evaluator
            self.ensemble[i].performance_evaluator.add_result(y[0], y_pred[0], sample_weight[0])

            # Train using random subspaces without resampling,
            # i.e. all instances are used for training.
            if self.training_method == self._TRAIN_RANDOM_SUBSPACES:
                self.ensemble[i].partial_fit(X=X, y=y, classes=classes,
                                             sample_weight=np.asarray([1.]),
                                             n_samples_seen=self._n_samples_seen,
                                             random_state=self._random_state)
            # Train using random patches or resampling,
            # thus we simulate online bagging with Poisson(lambda=...)
            else:
                k = self._random_state.poisson(lam=self.lam)
                if k > 0:
                    self.ensemble[i].partial_fit(X=X, y=y, classes=classes,
                                                 sample_weight=np.asarray([k]),
                                                 n_samples_seen=self._n_samples_seen,
                                                 random_state=self._random_state)

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
        n_samples, n_features = get_dimensions(X)

        if self.ensemble is None:
            self._init_ensemble(n_features=n_features)
            return np.zeros(n_samples)

        y_proba = self.predict_proba(X)
        y_pred = np.argmax(y_proba, axis=1)
        return y_pred

    def predict_proba(self, X):
        """ Estimate the probability of X belonging to each class-labels.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Samples one wants to predict the class probabilities for.

        Returns
        -------
        A numpy.ndarray of shape (n_samples, n_labels), in which each outer
        entry is associated with the X entry of the same index. And where the
        list in index [i] contains len(self.target_values) elements, each of
        which represents the probability that the i-th sample of X belongs to
        a certain class-label.

        """
        n_samples, n_features = get_dimensions(X)
        y_proba = []

        if self.ensemble is None:
            self._init_ensemble(n_features=n_features)
            return np.zeros(n_samples)

        for i in range(n_samples):
            y_proba.append(self._predict_proba(np.asarray([X[i]])))
        return np.asarray(y_proba)

    def _predict_proba(self, X):
        y_proba = np.asarray([0.])

        for i in range(len(self.ensemble)):
            y_proba_temp = self.ensemble[i].predict_proba(X)
            if np.sum(y_proba_temp) > 0.0:
                y_proba_temp = normalize(y_proba_temp, norm='l1')[0].copy()
                acc = self.ensemble[i].performance_evaluator.accuracy_score()
                if not self.disable_weighted_vote and acc > 0.0:
                    y_proba_temp *= acc
                # Check array length consistency
                if len(y_proba_temp) != len(y_proba):
                    if len(y_proba_temp) > len(y_proba):
                        y_proba.resize((len(y_proba_temp), ), refcheck=False)
                    else:
                        y_proba_temp.resize((len(y_proba), ), refcheck=False)
                # Add values
                y_proba += y_proba_temp
        return y_proba

    def _init_ensemble(self, n_features: int):
        # Select the size of k, which depends on 2 parameters:
        # subspace_size and subspace_mode
        k = self.subspace_size

        if self.training_method != self._TRAIN_RESAMPLING:
            # This only applies to subspaces and random patches options
            if self.subspace_mode == self._FEATURES_SQRT:
                k = int(np.round(np.sqrt(n_features)) + 1)
            elif self.subspace_mode == self._FEATURES_SQRT_INV:
                k = n_features - int(np.round(np.sqrt(n_features)) + 1)
            elif self.subspace_mode == self._FEATURES_PERCENT:
                percent = (100. + k) / 100. if k < 0 else k / 100.
                k = int(np.round(n_features * percent))
                if k < 2:
                    k = int(np.round(n_features * percent)) + 1
            # else: do nothing (k = m)
            if k < 0:
                # k is negative, calculate M - k
                k = n_features + k

        # Generate subspaces. The subspaces is a 2D matrix of shape
        # (n_estimators, k) where each row contains the k feature indices
        # to be used by each estimator.
        if self.training_method == self._TRAIN_RANDOM_SUBSPACES or \
                self.training_method == self._TRAIN_RANDOM_PATCHES:
            if k != 0 and k < n_features:
                # For low dimensionality it is better to avoid more than
                # 1 classifier with the same subspace, thus we generate all
                # possible combinations of subsets of features and select
                # without replacement.
                # n_features is the total number of features and k is the
                # actual size of the subspaces.
                if n_features <= 20 or k < 2:
                    if k == 1 and n_features > 2:
                        k = 2
                    # Generate all possible combinations of size k
                    self._subspaces = get_all_k_combinations(k, n_features)
                    # Increase the subspaces to match the ensemble size
                    # (if required)
                    i = 0
                    while len(self._subspaces) < self.n_estimators:
                        i = 0 if i == len(self._subspaces) else i
                        np.vstack((self._subspaces, self._subspaces[i]))
                        i += 1
                # For high dimensionality we can't generate all combinations
                # as it is too expensive (memory). On top of that, the chance
                # of repeating a subspace is lower, so we can just randomly
                # generate subspaces without worrying about repetitions.
                else:
                    self._subspaces = get_random_k_combinations(k, n_features,
                                                                self.n_estimators,
                                                                self._random_state)

            # k == 0 or k > n_features (subspace size is larger than the
            # number of features), then default to re-sampling
            else:
                self.training_method = self._TRAIN_RESAMPLING

        # Reset the base estimator for safety.
        self.base_estimator.reset()

        # Initialize ensemble members
        self._init_ensemble_members()

    def _init_ensemble_members(self):
        # Create empty ensemble:
        base_learner_class = self._base_learner_class
        self.ensemble = []   # type: List[base_learner_class]

        performance_evaluator = self._base_performance_evaluator

        subspace_indexes = np.arange(self.n_estimators)
        if self.training_method == self._TRAIN_RANDOM_PATCHES or \
                self.training_method == self._TRAIN_RANDOM_SUBSPACES:
            # Shuffle indexes that match subspaces with members of the ensemble
            self._random_state.shuffle(subspace_indexes)
        for i in range(self.n_estimators):
            # When self.training_method == self._TRAIN_RESAMPLING
            features_indexes = None
            # Otherwise set feature indexes
            if self.training_method == self._TRAIN_RANDOM_PATCHES or \
                    self.training_method == self._TRAIN_RANDOM_SUBSPACES:
                features_indexes = self._subspaces[subspace_indexes[i]]
            self.ensemble.append(base_learner_class(
                idx_original=i,
                base_estimator=clone(self.base_estimator),
                performance_evaluator=deepcopy(performance_evaluator),
                created_on=self._n_samples_seen,
                disable_background_learner=self.disable_background_learner,
                disable_drift_detector=self.disable_drift_detection,
                drift_detection_method=self.drift_detection_method,
                warning_detection_method=self.warning_detection_method,
                drift_detection_criteria=self.drift_detection_criteria,
                is_background_learner=False,
                feature_indexes=features_indexes,
                nominal_attributes=self.nominal_attributes,
                random_state=self._random_state))

    def reset(self):
        self.ensemble = None
        self._n_samples_seen = 0
        self._random_state = check_random_state(self.random_state)


class StreamingRandomPatchesBaseLearner:
    """
    Class representing the base learner of StreamingRandomPatchesClassifier.
    """
    def __init__(self,
                 idx_original,
                 base_estimator,
                 performance_evaluator,
                 created_on,
                 disable_background_learner,
                 disable_drift_detector,
                 drift_detection_method,
                 warning_detection_method,
                 drift_detection_criteria,
                 is_background_learner,
                 feature_indexes=None,
                 nominal_attributes=None,
                 random_state=None):
        self.idx_original = idx_original
        self.created_on = created_on
        self.base_estimator = base_estimator
        self.performance_evaluator = performance_evaluator

        # Store current model subspace representation of the original instances
        self.feature_indexes = feature_indexes

        # Drift detection
        self.disable_background_learner = disable_background_learner
        self.disable_drift_detector = disable_drift_detector
        self.drift_detection_method = clone(drift_detection_method)   # type: DriftDetector
        self.warning_detection_method = clone(warning_detection_method)   # type: DriftDetector
        self.drift_detection_criteria = drift_detection_criteria

        # Background learner
        self.is_background_learner = is_background_learner

        # Statistics
        self.n_drifts_detected = 0
        self.n_drifts_induced = 0
        self.n_warnings_detected = 0
        self.n_warnings_induced = 0

        # Background learner
        self._background_learner = None   # type: Optional[StreamingRandomPatchesBaseLearner]
        self._background_learner_class = StreamingRandomPatchesBaseLearner

        # Nominal attributes
        self.nominal_attributes = nominal_attributes
        self._set_nominal_attributes = self._can_set_nominal_attributes()

        # Random_state
        self.random_state = random_state
        self._random_state = check_random_state(self.random_state)

    def partial_fit(self, X: np.ndarray, y: np.ndarray, classes: list, sample_weight: np.ndarray,
                    n_samples_seen: int, random_state: np.random):
        n_features_total = get_dimensions(X)[1]
        if self.feature_indexes is not None:
            # Select the subset of features to use
            X_subset = np.asarray([X[0][self.feature_indexes]])
            if self._set_nominal_attributes and hasattr(self.base_estimator, 'nominal_attributes'):
                self.base_estimator.nominal_attributes = \
                    self._remap_nominal_attributes(self.feature_indexes, self.nominal_attributes)
                self._set_nominal_attributes = False
        else:
            # Use all features
            X_subset = X

        self.base_estimator.partial_fit(X=X_subset, y=y,
                                        classes=classes,
                                        sample_weight=sample_weight)
        correctly_classifies = self.base_estimator.predict(X_subset)[0] == y
        if self._background_learner:
            # Note: Pass the original instance X so features are correctly
            # selected at the beginning of partial_fit
            self._background_learner.partial_fit(X=X, y=y,
                                                 classes=classes,
                                                 sample_weight=sample_weight,
                                                 n_samples_seen=n_samples_seen,
                                                 random_state=random_state)

        if not self.disable_drift_detector and not self.is_background_learner:
            # Check for warnings only if the background learner is active
            if not self.disable_background_learner:
                # Update the warning detection method
                self.warning_detection_method.update(0 if correctly_classifies else 1)
                # Check if there was a change
                if self.warning_detection_method.change_detected:
                    self.n_warnings_detected += 1
                    self._trigger_warning(n_features=n_features_total,
                                          n_samples_seen=n_samples_seen,
                                          random_state=random_state)

            # ===== Drift detection =====
            # Update the drift detection method
            self.drift_detection_method.update(0 if correctly_classifies else 1)
            # Check if the was a change
            if self.drift_detection_method.change_detected:
                self.n_drifts_detected += 1
                # There was a change, reset the model
                self.reset(n_features=n_features_total, n_samples_seen=n_samples_seen,
                           random_state=random_state)

    def predict_proba(self, X):
        if self.feature_indexes is not None:
            # Select the subset of features to use
            X_subset = np.asarray([X[0][self.feature_indexes]])
        else:
            # Use all features
            X_subset = X

        return self.base_estimator.predict_proba(X_subset)

    def _trigger_warning(self, n_features, n_samples_seen: int, random_state: np.random):
        background_base_estimator = clone(self.base_estimator)
        background_base_estimator.reset()

        background_performance_evaluator = deepcopy(self.performance_evaluator)
        background_performance_evaluator.reset()

        feature_indexes = self._reset_subset(n_features=n_features, random_state=random_state)

        self._background_learner = self._background_learner_class(
            idx_original=self.idx_original,
            base_estimator=background_base_estimator,
            performance_evaluator=background_performance_evaluator,
            created_on=n_samples_seen,
            disable_background_learner=self.disable_background_learner,
            disable_drift_detector=self.disable_drift_detector,
            drift_detection_method=self.drift_detection_method,
            warning_detection_method=self.warning_detection_method,
            drift_detection_criteria=self.drift_detection_criteria,
            is_background_learner=True,
            feature_indexes=feature_indexes,
            nominal_attributes=self.nominal_attributes,
            random_state=self._random_state
        )

        # Hard-reset the warning method
        self.warning_detection_method = clone(self.warning_detection_method)

    def _reset_subset(self, n_features: int, random_state: np.random):
        feature_indexes = None
        if self.feature_indexes is not None:
            k = len(self.feature_indexes)
            feature_indexes = random_state.choice(range(n_features), k, replace=False)
        return feature_indexes

    def reset(self, n_features: int, n_samples_seen: int, random_state: np.random):
        if not self.disable_background_learner and self._background_learner is not None:
            self.base_estimator = self._background_learner.base_estimator
            self.drift_detection_method = self._background_learner.drift_detection_method
            self.warning_detection_method = self._background_learner.warning_detection_method
            self.performance_evaluator = self._background_learner.performance_evaluator
            self.performance_evaluator.reset()
            self.created_on = self._background_learner.created_on
            self.feature_indexes = self._background_learner.feature_indexes
            self._background_learner = None
        else:
            self.base_estimator.reset()
            self.performance_evaluator.reset()
            self.created_on = n_samples_seen
            self.drift_detection_method = clone(self.drift_detection_method)
            self.feature_indexes = self._reset_subset(n_features, random_state)
            self._set_nominal_attributes = self._can_set_nominal_attributes()

    @staticmethod
    def _remap_nominal_attributes(sel_features: np.ndarray, nominal_attributes: list) -> list:
        remapped_idx = []
        for i, idx in enumerate([i for i in sel_features]):
            if idx in nominal_attributes:
                remapped_idx.append(i)
        return remapped_idx if len(remapped_idx) > 0 else None

    def _can_set_nominal_attributes(self):
        return True if (self.nominal_attributes is not None and len(self.nominal_attributes) > 0) \
            else False


def _get_all_k_combinations_rec(offset: int, k: int, combination: deque, original_size: int,
                                combinations: deque):
    """ Recursive function to generate all k-combinations. """
    if k == 0:
        combinations.append(deepcopy(combination))
        return

    for i in range(offset, original_size - k + 1, 1):
        combination.append(i)
        _get_all_k_combinations_rec(i + 1, k - 1, combination, original_size, combinations)
        combination.pop()


def get_all_k_combinations(k: int, n_items: int) -> np.ndarray:
    """ Generates all k-combinations from n_features

    Parameters
    ----------
    k: int
        Number of items per combination
    n_items
        Total number of items

    Returns
    -------
    np.ndarray
        2D array containing all k-combinations

    """
    combinations = deque()
    combination = deque()
    _get_all_k_combinations_rec(0, k, combination, n_items, combinations)
    return np.array(combinations)


def get_random_k_combinations(k: int, n_items: int, n_combinations: int,
                              random_state: np.random) -> np.ndarray:
    """ Gets random k-combinations from n_features

    Parameters
    ----------
    k: int
        Number of items per combination
    n_items
        Total number of items
    n_combinations: int
        Number of combinations
    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    np.ndarray
        2D array containing all k-combinations

    """
    return np.array([random_state.choice(range(n_items), k, replace=False)
                     for _ in range(n_combinations)])
