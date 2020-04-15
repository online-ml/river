from copy import deepcopy
import math

import numpy as np

from skmultiflow.core import BaseSKMObject, RegressorMixin, MetaEstimatorMixin
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
from skmultiflow.drift_detection import ADWIN
from skmultiflow.trees.hoeffding_tree_regressor import HoeffdingTreeRegressor
from skmultiflow.metrics.measure_collection import RegressionMeasurements
from skmultiflow.utils import get_dimensions, check_random_state


class AdaptiveRandomForestRegressor(BaseSKMObject,
                                    RegressorMixin,
                                    MetaEstimatorMixin):
    """Adaptive Random Forest estimator.

    Parameters
    ----------
    n_estimators: int, optional (default=10)
        Number of trees in the ensemble.

    max_features : int, float, string or None,\
                   optional (default="auto")
        Max number of attributes for each node split.
        - If int, then consider ``max_features`` features at each split.
        - If float, then ``max_features`` is a percentage and
          ``int(max_features * n_features)`` features are considered at
          each split.
        - If "auto", then ``max_features=sqrt(n_features)``.
        - If "sqrt", then ``max_features=sqrt(n_features)``
          (same as "auto").
        - If "log2", then ``max_features=log2(n_features)``.
        - If None, then ``max_features=n_features``.

    lambda_value: int, optional (default=6)
        The lambda value for bagging (lambda=6 corresponds to Leverage
        Bagging).

    drift_detection_method: BaseDriftDetector or None,
                            optional (default=ADWIN(0.001))
        Drift Detection method. Set to None to disable Drift detection.

    warning_detection_method: BaseDriftDetector or None,
                              default(ADWIN(0.01))
        Warning Detection method. Set to None to disable warning
        detection.

    max_byte_size: int, optional (default=33554432)
        (`HoeffdingTreeRegressor` parameter)
        Maximum memory consumed by the tree.

    memory_estimate_period: int, optional (default=2000000)
        (`HoeffdingTreeRegressor` parameter)
        Number of instances between memory consumption checks.

    grace_period: int, optional (default=50)
        (`HoeffdingTreeRegressor` parameter)
        Number of instances a leaf should observe between split
        attempts.

    split_confidence: float, optional (default=0.01)
        (`HoeffdingTreeRegressor` parameter)
        Allowed error in split decision, a value closer to 0 takes
        longer to decide.

    tie_threshold: float, optional (default=0.05)
        (`HoeffdingTreeRegressor` parameter)
        Threshold below which a split will be forced to break ties.

    binary_split: bool, optional (default=False)
        (`HoeffdingTreeRegressor` parameter)
        If True, only allow binary splits.

    stop_mem_management: bool, optional (default=False)
        (`HoeffdingTreeRegressor` parameter)
        If True, stop growing as soon as memory limit is hit.

    remove_poor_atts: bool, optional (default=False)
        (`HoeffdingTreeRegressor` parameter)
        If True, disable poor attributes.

    no_preprune: bool, optional (default=False)
        (`HoeffdingTreeRegressor` parameter)
        If True, disable pre-pruning.

    leaf_prediction: string, optional (default='perceptron')
        (`HoeffdingTreeRegressor` parameter)
        Prediction mechanism used at leafs.
        - 'mean' - Target mean
        - 'perceptron' - Perceptron

    nominal_attributes: list, optional (default=None)
        (`HoeffdingTreeRegressor` parameter)
        List of Nominal attributes. If emtpy, then assume that all
        attributes are numerical.

    learning_ratio_perceptron: float (default=0.02)
        (`HoeffdingTreeRegressor` parameter)
        The learning rate of the perceptron.

    learning_ratio_decay: float (default=0.001)
        (`HoeffdingTreeRegressor` parameter)
        Decay multiplier for the learning rate of the perceptron

    learning_ratio_const: Bool (default=True)
        (`HoeffdingTreeRegressor` parameter)
        If False the learning ratio will decay with the number of
        examples seen.

    random_state: int, RandomState instance or None,
                       optional (default=None)
        If int, random_state is the seed used by the random number
        generator;
        If RandomState instance, random_state is the random number
        generator;
        If None, the random number generator is the RandomState
        instance used by `np.random`. Used when leaf_prediction is
        'perceptron'.

    Notes
    -----
    The 3 most important aspects of Adaptive Random Forest [1]_ are:
    (1) inducing diversity through re-sampling;
    (2) inducing diversity through randomly selecting subsets of
        features for node splits
        (see skmultiflow.classification.trees.arf_hoeffding_tree);
    (3) drift detectors per base tree, which cause selective resets
        in response to drifts.
    It also allows training background trees, which start training if
    a warning is detected and replace the active tree if the warning
    escalates to a drift.

    References
    ----------
    .. [1] Gomes, H.M., Bifet, A., Read, J., Barddal, J.P.,
       Enembreck, F., Pfharinger, B., Holmes, G. and Abdessalem,
       T., 2017.
       Adaptive random forests for evolving data stream classification.
       Machine Learning, 106(9-10), pp.1469-1495.
       [2] Gomes, H.M., Barddal, J.P., Boiko, L.E., Bifet, A., 2018
       Adaptive random forests for data stream regression.
       ESANN 2018.
    """

    def __init__(self,
                 # Forest parameters
                 n_estimators=10,
                 max_features='auto',
                 lambda_value=6,
                 drift_detection_method: BaseDriftDetector=ADWIN(0.001),
                 warning_detection_method: BaseDriftDetector=ADWIN(0.01),
                 # Tree parameters
                 max_byte_size=33554432,
                 memory_estimate_period=1000000,
                 grace_period=200,
                 split_confidence=0.0000001,
                 tie_threshold=0.05,
                 binary_split=False,
                 stop_mem_management=False,
                 remove_poor_atts=False,
                 no_preprune=False,
                 leaf_prediction='perceptron',
                 nominal_attributes=None,
                 learning_ratio_perceptron=0.02,
                 learning_ratio_decay=0.001,
                 learning_ratio_const=True,
                 random_state=None):
        """ AdaptiveRandomForestRegressor class constructor. """
        super().__init__()
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.lambda_value = lambda_value
        if isinstance(drift_detection_method, BaseDriftDetector):
            self.drift_detection_method = drift_detection_method
        else:
            self.drift_detection_method = None
        if isinstance(warning_detection_method, BaseDriftDetector):
            self.warning_detection_method = warning_detection_method
        else:
            self.warning_detection_method = None
        self.instances_seen = 0
        self.ensemble = None
        self.random_state = random_state
        # This is the actual random_state object used
        self._random_state = check_random_state(self.random_state)

        # Hoeffding Tree Regressor configuration
        self.max_byte_size = max_byte_size
        self.memory_estimate_period = memory_estimate_period
        self.grace_period = grace_period
        self.split_confidence = split_confidence
        self.tie_threshold = tie_threshold
        self.binary_split = binary_split
        self.stop_mem_management = stop_mem_management
        self.remove_poor_atts = remove_poor_atts
        self.no_preprune = no_preprune
        self.leaf_prediction = leaf_prediction
        self.nominal_attributes = nominal_attributes
        self.learning_ratio_perceptron = learning_ratio_perceptron
        self.learning_ratio_decay = learning_ratio_decay
        self.learning_ratio_const = learning_ratio_const

    def partial_fit(self, X, y):
        """ Partially (incrementally) fit the model.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.

        y: numpy.ndarray of shape (n_samples)
            An array-like with the target values of all samples in X.

        Returns
        -------
        self

        """
        if y is None:
            return self
        if self.ensemble is None:
            self.init_ensemble(X)

        for i in range(get_dimensions(X)[0]):
            self.instances_seen += 1

            for learner in self.ensemble:
                y_predicted = learner.predict(X[i:i+1])
                learner.evaluator.add_result(y_predicted, y)
                k = self._random_state.poisson(self.lambda_value)
                if k > 0:
                    learner.partial_fit(X[i:i+1], y[i:i+1],
                                        sample_weight=np.asarray([k]),
                                        instances_seen=self.instances_seen)

        return self

    def predict(self, X):
        """ Predict target values for the passed data.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The set of data samples for which to predict the target
            value.

        Returns
        -------
        A numpy.ndarray with all the predictions for the samples in X.

        """
        predictions = np.zeros(get_dimensions(X)[0])
        if self.ensemble is None:
            self.init_ensemble(X)
        for learner in self.ensemble:
            predictions += learner.predict(X)
        predictions /= self.n_estimators
        return predictions

    def predict_proba(self, X):
        """Not implemented for this method."""
        raise NotImplementedError

    def reset(self):
        """Reset ARF."""
        self.ensemble = None
        self.max_features = 0
        self.instances_seen = 0
        self._random_state = check_random_state(self.random_state)

    def init_ensemble(self, X):
        self._set_max_features(get_dimensions(X)[1])

        self.ensemble = \
            [ARFBaseLearner(
                index_original=i,
                estimator=HoeffdingTreeRegressor(
                    max_byte_size=self.max_byte_size,
                    memory_estimate_period=self.memory_estimate_period,
                    grace_period=self.grace_period,
                    split_confidence=self.split_confidence,
                    tie_threshold=self.tie_threshold,
                    binary_split=self.binary_split,
                    stop_mem_management=self.stop_mem_management,
                    remove_poor_atts=self.remove_poor_atts,
                    no_preprune=self.no_preprune,
                    leaf_prediction=self.leaf_prediction,
                    nominal_attributes=self.nominal_attributes,
                    learning_ratio_perceptron=self.learning_ratio_perceptron,
                    learning_ratio_decay=self.learning_ratio_decay,
                    learning_ratio_const=self.learning_ratio_const,
                    random_state=self.random_state),
                instances_seen=self.instances_seen,
                drift_detection_method=self.drift_detection_method,
                warning_detection_method=self.warning_detection_method,
                is_background_learner=False)
            for i in range(self.n_estimators)]

    def _set_max_features(self, n):
        if self.max_features == 'auto' or self.max_features == 'sqrt':
            self.max_features = round(math.sqrt(n))
        elif self.max_features == 'log2':
            self.max_features = round(math.log2(n))
        elif isinstance(self.max_features, int):
            # Consider 'max_features' features at each split.
            pass
        elif isinstance(self.max_features, float):
            # Consider 'max_features' as a percentage
            if self.max_feature <= 0 or self.max_feature > 1:
                raise ValueError('Invalid max_feature: {}'.format(max_feature))
            self.max_features = int(self.max_features * n)
        elif self.max_features is None:
            self.max_features = n
        else:
            # Default to "auto"
            self.max_features = round(math.sqrt(n))
        # Sanity checks
        # max_features is negative, use max_features + n
        if self.max_features < 0:
            self.max_features += n
        # max_features <= 0
        # (m can be negative if max_features is negative
        # and abs(max_features) > n)
        # use max_features = 1
        if self.max_features <= 0:
            self.max_features = 1
        # max_features > n, then use n
        if self.max_features > n:
            self.max_features = n

    @staticmethod
    def is_randomizable():
        return True


class ARFBaseLearner(BaseSKMObject):
    """ARF Base Learner class.

    Parameters
    ----------
    index_original: int
        Tree index within the ensemble.

    estimator: HoeffdingTreeRegressor
        Tree estimator.

    instances_seen: int
        Number of instances seen by the tree.

    drift_detection_method: BaseDriftDetector
        Drift Detection method.

    warning_detection_method: BaseDriftDetector
        Warning Detection method.

    is_background_learner: bool
        True if the tree is a background learner.

    Notes
    -----
    Inner class that represents a single tree member of the forest.
    Contains analysis information, such as the numberOfDriftsDetected.

    """

    def __init__(self,
                 index_original,
                 estimator: HoeffdingTreeRegressor,
                 instances_seen,
                 drift_detection_method: BaseDriftDetector,
                 warning_detection_method: BaseDriftDetector,
                 is_background_learner):
        self.index_original = index_original
        self.estimator = estimator
        self.created_on = instances_seen
        self.is_background_learner = is_background_learner
        self.evaluator_method = RegressionMeasurements

        # Drift and warning
        self.drift_detection_method = drift_detection_method
        self.warning_detection_method = warning_detection_method

        self.last_drift_on = 0
        self.last_warning_on = 0
        self.n_drifts_detected = 0
        self.n_warnings_detected = 0

        self.drift_detection = None
        self.warning_detection = None
        self.background_learner = None
        self._use_drift_detector = False
        self._use_background_learner = False

        self.evaluator = self.evaluator_method()

        # Initialize drift and warning detectors
        if drift_detection_method is not None:
            self._use_drift_detector = True
            self.drift_detection = deepcopy(drift_detection_method)

        if warning_detection_method is not None:
            self._use_background_learner = True
            self.warning_detection = deepcopy(warning_detection_method)

    def reset(self, instances_seen):
        if self._use_background_learner and self.background_learner is not None:
            self.estimator = self.background_learner.estimator
            self.warning_detection = self.background_learner.warning_detection
            self.drift_detection = self.background_learner.drift_detection
            self.evaluator_method = self.background_learner.evaluator_method
            self.created_on = self.background_learner.created_on
            self.background_learner = None
        else:
            self.estimator.reset()
            self.created_on = instances_seen
            self.drift_detection.reset()
        self.evaluator = self.evaluator_method()

    def partial_fit(self, X, y, sample_weight, instances_seen):
        self.estimator.partial_fit(X, y, sample_weight=sample_weight)

        if self.background_learner:
            self.background_learner.estimator.partial_fit(X, y,
                sample_weight=sample_weight)

        if self._use_drift_detector and not self.is_background_learner:
            predicted_value = self.estimator.predict(X)
            # Check for warning only if use_background_learner is active
            if self._use_background_learner:
                self.warning_detection.add_element(predicted_value)
                # Check if there was a change
                if self.warning_detection.detected_change():
                    self.last_warning_on = instances_seen
                    self.n_warnings_detected += 1
                    # Create a new background tree estimator
                    background_learner = self.estimator.new_instance()
                    # Create a new background learner
                    self.background_learner = \
                        ARFBaseLearner(self.index_original,
                                       background_learner,
                                       instances_seen,
                                       self.drift_detection_method,
                                       self.warning_detection_method,
                                       True)
                    # Update the warning detection object for the current object
                    # (this effectively resets changes made to the object
                    # while it was still a bkg learner).
                    self.warning_detection.reset()

            # Update the drift detection
            self.drift_detection.add_element(predicted_value)

            # Check if there was a change
            if self.drift_detection.detected_change():
                self.last_drift_on = instances_seen
                self.n_drifts_detected += 1
                self.reset(instances_seen)

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)
