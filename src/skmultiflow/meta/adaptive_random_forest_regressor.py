from copy import deepcopy
import math

import numpy as np

from skmultiflow.core import BaseSKMObject, RegressorMixin
from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
from skmultiflow.drift_detection import ADWIN
from skmultiflow.trees.arf_hoeffding_tree_regressor import ARFHoeffdingTreeRegressor
from skmultiflow.metrics.measure_collection import RegressionMeasurements
from skmultiflow.utils import get_dimensions, check_random_state


class AdaptiveRandomForestRegressor(RegressorMixin, AdaptiveRandomForestClassifier):
    """Adaptive Random Forest regressor.

    Parameters
    ----------
    n_estimators: int, optional (default=10)
        Number of trees in the ensemble.

    max_features : int, float, str or None, optional (default="auto")
        | Max number of attributes for each node split.
        | - If int, then consider ``max_features`` features at each split.
        | - If float, then ``max_features`` is a percentage and \
          ``int(max_features * n_features)`` features are considered at each split.
        | - If "auto", then ``max_features=sqrt(n_features)``.
        | - If "sqrt", then ``max_features=sqrt(n_features)`` (same as "auto").
        | - If "log2", then ``max_features=log2(n_features)``.
        | - If None, then ``max_features=n_features``.

    lambda_value: int, optional (default=6)
        The lambda value for bagging (lambda=6 corresponds to Leverage Bagging).

    aggregation_method: str, optional (default='median')
        | The method to use to aggregate predictions in the ensemble.
        | - 'mean'
        | - 'median'

    weighted_vote_strategy: str or None, optional (default=None)
        | Metric used to weight individual tree's responses when aggregating them. \
        Only used when ``aggregation_method='mean'``. Possible values are:
        | - None: Do not assign weights to individual tree's predictions. \
          Use the arithmetic mean instead.
        | - 'mse': Weight predictions using trees' Mean Square Error
        | - 'mae': Weight predictions using trees' Mean Absolute Error

    drift_detection_method: BaseDriftDetector or None, optional (default=ADWIN(0.001))
        Drift Detection method. Set to None to disable Drift detection.

    warning_detection_method: BaseDriftDetector or None, default(ADWIN(0.01))
        Warning Detection method. Set to None to disable warning detection.

    drift_detection_criteria: str, optional (default='mse')
        | The criteria used to track drifts.
        | - 'mse' - Mean Square Error
        | - 'mae' - Mean Absolute Error
        | - 'predictions' - predicted target values

    max_byte_size: int, optional (default=1048576000)
        (`ARFHoeffdingTreeRegressor` parameter)
        Maximum memory consumed by the tree.

    memory_estimate_period: int, optional (default=2000000)
        (`ARFHoeffdingTreeRegressor` parameter)
        Number of instances between memory consumption checks.

    grace_period: int, optional (default=50)
        (`ARFHoeffdingTreeRegressor` parameter)
        Number of instances a leaf should observe between split
        attempts.

    split_confidence: float, optional (default=0.01)
        (`ARFHoeffdingTreeRegressor` parameter)
        Allowed error in split decision, a value closer to 0 takes
        longer to decide.

    tie_threshold: float, optional (default=0.05)
        (`ARFHoeffdingTreeRegressor` parameter)
        Threshold below which a split will be forced to break ties.

    binary_split: bool, optional (default=False)
        (`ARFHoeffdingTreeRegressor` parameter)
        If True, only allow binary splits.

    stop_mem_management: bool, optional (default=False)
        (`ARFHoeffdingTreeRegressor` parameter)
        If True, stop growing as soon as memory limit is hit.

    remove_poor_atts: bool, optional (default=False)
        (`ARFHoeffdingTreeRegressor` parameter)
        If True, disable poor attributes.

    no_preprune: bool, optional (default=False)
        (`ARFHoeffdingTreeRegressor` parameter)
        If True, disable pre-pruning.

    leaf_prediction: str, optional (default='perceptron')
        | (`ARFHoeffdingTreeRegressor` parameter) \
        Prediction mechanism used at leafs.
        | - 'mean' - Target mean
        | - 'perceptron' - Perceptron

    nominal_attributes: list, optional (default=None)
        (`ARFHoeffdingTreeRegressor` parameter)
        List of Nominal attributes. If emtpy, then assume that all
        attributes are numerical.

    learning_ratio_perceptron: float (default=0.1)
        (`ARFHoeffdingTreeRegressor` parameter)
        The learning rate of the perceptron.

    learning_ratio_decay: float (default=0.001)
        (`ARFHoeffdingTreeRegressor` parameter)
        Decay multiplier for the learning rate of the perceptron

    learning_ratio_const: Bool (default=True)
        (`ARFHoeffdingTreeRegressor` parameter)
        If False the learning ratio will decay with the number of
        examples seen.

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by `np.random`.
        Used when leaf_prediction is 'perceptron'.

    Notes
    -----
    The 3 most important aspects of Adaptive Random Forest [1]_ are:
    (1) inducing diversity through re-sampling;
    (2) inducing diversity through randomly selecting subsets of features for node splits
    (see skmultiflow.trees.arf_hoeffding_tree);
    (3) drift detectors per base tree, which cause selective resets in response to drifts.
    It also allows training background trees, which start training if a warning is detected and
    replace the active tree if the warning escalates to a drift.

    Notice that this implementation is slightly different from the original algorithm proposed
    in [2]_. The HoeffdingTreeRegressor is used as base learner, instead of FIMT-DD. It also adds a
    new strategy to monitor the incoming data and check for concept drifts. The monitored data
    (either the trees' errors or their predictions) are centered and scaled (z-score normalization)
    to have zero mean and unit standard deviation. Transformed values are then again normalized in
    the [0, 1] range to fulfil ADWIN's requirements. We assume that the data subjected to the
    z-score normalization lies within the interval of the mean :math:`\\pm3\\sigma`, as it occurs
    in normal distributions.

    References
    ----------
    .. [1] Gomes, H.M., Bifet, A., Read, J., Barddal, J.P., Enembreck, F., Pfharinger, B.,
        Holmes, G. and Abdessalem, T., 2017. Adaptive random forests for evolving data stream
        classification. Machine Learning, 106(9-10), pp.1469-1495.

    .. [2] Gomes, H.M., Barddal, J.P., Boiko, L.E., Bifet, A., 2018. Adaptive random forests for
       data stream regression. ESANN 2018.

    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.data import RegressionGenerator
    >>> from skmultiflow.meta import AdaptiveRandomForestRegressor
    >>> import numpy as np
    >>>
    >>> # Setup a data stream
    >>> stream = RegressionGenerator(random_state=1, n_samples=200)
    >>> # Prepare stream for use
    >>>
    >>> # Setup the Adaptive Random Forest regressor
    >>> arf_reg = AdaptiveRandomForestRegressor(random_state=123456)
    >>>
    >>> # Auxiliary variables to control loop and track performance
    >>> n_samples = 0
    >>> max_samples = 200
    >>> y_pred = np.zeros(max_samples)
    >>> y_true = np.zeros(max_samples)
    >>>
    >>> # Run test-then-train loop for max_samples and while there is data
    >>> while n_samples < max_samples and stream.has_more_samples():
    >>>     X, y = stream.next_sample()
    >>>     y_true[n_samples] = y[0]
    >>>     y_pred[n_samples] = arf_reg.predict(X)[0]
    >>>     arf_reg.partial_fit(X, y)
    >>>     n_samples += 1
    >>>
    >>> # Display results
    >>> print('Adaptive Random Forest regressor example')
    >>> print('{} samples analyzed.'.format(n_samples))
    >>> print('Mean absolute error: {}'.format(np.mean(np.abs(y_true - y_pred))))

    """

    _MEAN = 'mean'
    _MEDIAN = 'median'
    _MAE = 'mae'
    _MSE = 'mse'
    _PREDICTIONS = 'predictions'

    def __init__(self,
                 # Forest parameters
                 n_estimators: int = 10,
                 max_features='auto',
                 aggregation_method: str = 'median',
                 weighted_vote_strategy: str = None,
                 lambda_value: int = 6,
                 drift_detection_method: BaseDriftDetector = ADWIN(0.001),
                 warning_detection_method: BaseDriftDetector = ADWIN(0.01),
                 drift_detection_criteria: str = 'mse',
                 # Tree parameters
                 max_byte_size: int = 1048576000,
                 memory_estimate_period: int = 2000000,
                 grace_period: int = 50,
                 split_confidence: float = 0.01,
                 tie_threshold: float = 0.05,
                 binary_split: bool = False,
                 stop_mem_management: bool = False,
                 remove_poor_atts: bool = False,
                 no_preprune: bool = False,
                 leaf_prediction: str = 'perceptron',
                 nominal_attributes: list = None,
                 learning_ratio_perceptron: float = 0.1,
                 learning_ratio_decay: float = 0.001,
                 learning_ratio_const: bool = True,
                 random_state=None):
        super().__init__(n_estimators=n_estimators,
                         max_features=max_features,
                         lambda_value=lambda_value,
                         drift_detection_method=drift_detection_method,
                         warning_detection_method=warning_detection_method,
                         # Tree parameters
                         max_byte_size=max_byte_size,
                         memory_estimate_period=memory_estimate_period,
                         grace_period=grace_period,
                         split_confidence=split_confidence,
                         tie_threshold=tie_threshold,
                         binary_split=binary_split,
                         stop_mem_management=stop_mem_management,
                         remove_poor_atts=remove_poor_atts,
                         no_preprune=no_preprune,
                         leaf_prediction=leaf_prediction,
                         nominal_attributes=nominal_attributes,
                         random_state=random_state)

        self.learning_ratio_perceptron = learning_ratio_perceptron
        self.learning_ratio_decay = learning_ratio_decay
        self.learning_ratio_const = learning_ratio_const

        if weighted_vote_strategy in [self._MSE, self._MAE, None]:
            self.weighted_vote_strategy = weighted_vote_strategy
        else:
            raise ValueError('Invalid weighted vote strategy: {}'.format(weighted_vote_strategy))

        if aggregation_method in [self._MEAN, self._MEDIAN]:
            self.aggregation_method = aggregation_method
        else:
            raise ValueError('Invalid aggregation method: {}'.format(aggregation_method))

        if drift_detection_criteria in [self._MSE, self._MAE, self._PREDICTIONS]:
            self.drift_detection_criteria = drift_detection_criteria
        else:
            raise ValueError('Invalid drift detection criteria: {}'.
                             format(drift_detection_criteria))

    def partial_fit(self, X, y, sample_weight=None):
        """ Partially (incrementally) fit the model.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.

        y: numpy.ndarray of shape (n_samples)
            An array-like with the target values of all samples in X.

        sample_weight: (default=None)
            This parameter it is not used in AdaptiveRandomForestRegressor since the ensemble
            algorithm internally assign different weights to the incoming instances. Kept for
            method's signature compatibility purpose only.

        Returns
        -------
        self

        """
        if y is None:
            return self
        if self.ensemble is None:
            self._init_ensemble(X)

        for i in range(get_dimensions(X)[0]):
            self.instances_seen += 1

            for learner in self.ensemble:
                k = self._random_state.poisson(self.lambda_value)
                if k > 0:
                    learner.partial_fit(
                        X[i].reshape(1, -1), [y[i]], sample_weight=[k],
                        instances_seen=self.instances_seen
                    )

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
        predictions = np.zeros((self.n_estimators, get_dimensions(X)[0]))
        if self.ensemble is None:
            self._init_ensemble(X)

        for i, learner in enumerate(self.ensemble):
            predictions[i, :] = learner.predict(X)

        if self.aggregation_method == self._MEAN:
            if self.weighted_vote_strategy is not None:
                weights = np.asarray([learner.get_error() for learner in self.ensemble])
                sum_weights = weights.sum()

                if sum_weights != 0:
                    # The higher the error, the worse is the tree
                    weights = sum_weights - weights
                    # Normalize weights to sum up to 1
                    weights = weights / weights.sum()
                    return np.average(predictions, weights=weights, axis=0)

            return predictions.mean(axis=0)
        elif self.aggregation_method == self._MEDIAN:
            return np.median(predictions, axis=0)
        else:
            np.zeros(get_dimensions(X)[0])

    def predict_proba(self, X):
        """Not implemented for this method."""
        raise NotImplementedError

    def reset(self):
        """Reset ARFR."""

        # TODO: check whether this is enough

        self.ensemble = None
        self.max_features = 0
        self.instances_seen = 0
        self._random_state = check_random_state(self.random_state)

    def _init_ensemble(self, X):
        self._set_max_features(get_dimensions(X)[1])
        # Generate a different random seed per tree
        random_states = self._random_state.randint(0, 4294967295, size=self.n_estimators,
                                                   dtype='u8')
        self.ensemble = [
            ARFRegBaseLearner(
                index_original=i,
                estimator=ARFHoeffdingTreeRegressor(
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
                    max_features=self.max_features,
                    random_state=random_states[i]
                ),
                instances_seen=self.instances_seen,
                drift_detection_method=self.drift_detection_method,
                warning_detection_method=self.warning_detection_method,
                performance_metric=self.weighted_vote_strategy,
                drift_detection_criteria=self.drift_detection_criteria,
                is_background_learner=False
            ) for i in range(self.n_estimators)
        ]

    def _set_max_features(self, n):
        if self.max_features == 'auto' or self.max_features == 'sqrt':
            self.max_features = int(round(math.sqrt(n)))
        elif self.max_features == 'log2':
            self.max_features = int(round(math.log2(n)))
        elif isinstance(self.max_features, int):
            # Consider 'max_features' features at each split.
            pass
        elif isinstance(self.max_features, float):
            # Consider 'max_features' as a percentage
            if self.max_features <= 0 or self.max_features > 1:
                raise ValueError('Invalid max_features value: {}'.format(self.max_features))

            self.max_features = int(self.max_features * n)
        elif self.max_features is None:
            self.max_features = n
        else:
            # Default to "auto"
            self.max_features = int(round(math.sqrt(n)))
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


class ARFRegBaseLearner(BaseSKMObject):
    """ARF Base Learner class.

    Parameters
    ----------
    index_original: int
        Tree index within the ensemble.

    estimator: ARFHoeffdingTreeRegressor
        Tree estimator.

    instances_seen: int
        Number of instances seen by the tree.

    drift_detection_method: BaseDriftDetector
        Drift Detection method.

    warning_detection_method: BaseDriftDetector
        Warning Detection method.

    performance_metric: str
        Metric used to track trees performance within the ensemble.
        - 'mse': Mean Square Error
        - 'mae': Mean Absolute Error
        - None: Do not track tree's performance

    drift_detection_criteria: str
        The criteria used to track drifts.
            - 'mse' - Mean Square Error
            - 'mae' - Mean Absolute Error
            - 'predictions' - predicted target values

    is_background_learner: bool
        True if the tree is a background learner.

    Notes
    -----
    Inner class that represents a single tree member of the forest.
    Contains analysis information, such as the number of drifts detected.

    """

    _MAE = 'mae'
    _MSE = 'mse'

    def __init__(self,
                 index_original: int,
                 estimator: ARFHoeffdingTreeRegressor,
                 instances_seen: int,
                 drift_detection_method: BaseDriftDetector,
                 warning_detection_method: BaseDriftDetector,
                 performance_metric: str,
                 drift_detection_criteria: str,
                 is_background_learner):
        self.index_original = index_original
        self.estimator = estimator
        self.created_on = instances_seen
        self.is_background_learner = is_background_learner
        self.evaluator_method = RegressionMeasurements

        # Drift and warning
        self.drift_detection_method = drift_detection_method
        self.warning_detection_method = warning_detection_method

        self.performance_metric = performance_metric
        self.drift_detection_criteria = drift_detection_criteria

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

        # Normalization of info monitored by drift detectors (using Welford's algorithm)
        self._k = 0

    def _normalize_drift_input(self, drift_input):
        drift_input = drift_input[0]

        self._k += 1
        # Welford's algorithm update step
        if self._k == 1:
            self._pM = self._M = drift_input
            self._pS = 0

            return 0.0
        else:
            self._M = self._pM + (drift_input - self._pM) / self._k
            self._S = self._pS + (drift_input - self._pM) * (drift_input - self._M)

            # Save previously calculated values for the next iteration
            self._pM = self._M
            self._pS = self._S

            sd = math.sqrt(self._S / (self._k - 1))

            # Apply z-score normalization to drift input
            norm_input = (drift_input - self._M) / sd if sd > 0 else 0.0

            # Data with zero mean and unit variance -> (empirical rule) 99.73% of the values lie
            # between [mean - 3*sd, mean + 3*sd] (in a normal distribution): we assume this range
            # for the norm variable.
            # Hence, the values are assumed to be between [-3, 3] and we can apply the min-max norm
            # to cope with ADWIN's requirements
            return (norm_input + 3) / 6

    def reset(self, instances_seen):    # noqa
        if self._use_background_learner and self.background_learner is not None:
            self.estimator = self.background_learner.estimator
            self.evaluator = self.background_learner.evaluator
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

        # Reset normalization auxiliary variables
        self._k = 0

    def partial_fit(self, X, y, sample_weight, instances_seen):
        predicted_value = self.estimator.predict(X)
        # Monitor base learner performance
        self.evaluator.add_result(y[0], predicted_value[0])
        # Update learning model

        self.estimator.partial_fit(X, y, sample_weight=sample_weight)

        if self.background_learner:
            prediction_background = self.background_learner.estimator.predict(X)
            self.background_learner.evaluator.add_result(y[0], prediction_background[0])

            # Update background learner
            self.background_learner.estimator.partial_fit(X, y, sample_weight=sample_weight)

        if self._use_drift_detector and not self.is_background_learner:
            # Select which kind of data is going to be monitored
            if self.drift_detection_criteria == self._MSE:
                drift_input = (y - predicted_value) * (y - predicted_value)
            elif self.drift_detection_criteria == self._MAE:
                drift_input = abs(y[0] - predicted_value)
            else:  # predictions
                drift_input = predicted_value

            drift_input = self._normalize_drift_input(drift_input)

            # Check for warning only if use_background_learner is active
            if self._use_background_learner:
                self.warning_detection.add_element(drift_input)
                # Check if there was a change
                if self.warning_detection.detected_change():
                    self.last_warning_on = instances_seen
                    self.n_warnings_detected += 1

                    # Create a new background tree estimator
                    background_learner = self.estimator.new_instance()
                    # Create a new background learner
                    self.background_learner = ARFRegBaseLearner(
                        index_original=self.index_original,
                        estimator=background_learner,
                        instances_seen=instances_seen,
                        drift_detection_method=self.drift_detection_method,
                        warning_detection_method=self.warning_detection_method,
                        performance_metric=self.performance_metric,
                        drift_detection_criteria=self.drift_detection_criteria,
                        is_background_learner=True
                    )
                    # Update the warning detection object for the current object
                    # (this effectively resets changes made to the object
                    # while it was still a bkg learner).
                    self.warning_detection.reset()

            # Update the drift detection
            self.drift_detection.add_element(drift_input)

            # Check if there was a change
            if self.drift_detection.detected_change():
                self.last_drift_on = instances_seen
                self.n_drifts_detected += 1

                self.reset(instances_seen)

    def get_error(self):
        if self.performance_metric == self._MSE:
            return self.evaluator.get_mean_square_error()
        elif self.performance_metric == self._MAE:
            return self.evaluator.get_average_error()
        return self.evaluator.get_mean_square_error()  # Defaults to MSE

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        raise NotImplementedError
