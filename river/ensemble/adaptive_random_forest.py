import collections
import math
import typing
import copy

from river import base
from river.drift import ADWIN
from river.metrics import Accuracy
from river.metrics.base import MultiClassMetric
from river.tree.arf_hoeffding_tree_classifier import ARFHoeffdingTreeClassifier
from river.utils.skmultiflow_utils import check_random_state


class AdaptiveRandomForestClassifier(base.EnsembleMixin, base.Classifier):
    """Adaptive Random Forest classifier.

    The 3 most important aspects of Adaptive Random Forest [^1] are:
    1. inducing diversity through re-sampling;
    2. inducing diversity through randomly selecting subsets of features for
       node splits
    3. drift detectors per base tree, which cause selective resets in response
       to drifts.

    It also allows training background trees, which start training if a
    warning is detected and replace the active tree if the warning escalates
    to a drift.

    Parameters
    ----------
    n_models
        Number of trees in the ensemble.
    max_features
        Max number of attributes for each node split.<br/>
        - If `int`, then consider `max_features` at each split.<br/>
        - If `float`, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered per split.<br/>
        - If "sqrt", then `max_features=sqrt(n_features)`.<br/>
        - If "log2", then `max_features=log2(n_features)`.<br/>
        - If None, then ``max_features=n_features``.
    disable_weighted_vote
        If `True`, disables the weighted vote prediction.
    lambda_value
        The lambda value for bagging (lambda=6 corresponds to Leveraging Bagging).
    metric
        Metric used to track trees performance within the ensemble.
    drift_detector
        Drift Detection method. Set to None to disable Drift detection.
    warning_detector
        Warning Detection method. Set to None to disable warning detection.
    max_size
        (`ARFHoeffdingTreeClassifier` parameter)
        Maximum memory consumed by the tree.
    memory_estimate_period
        (`ARFHoeffdingTreeClassifier` parameter)
        Number of instances between memory consumption checks.
    grace_period
        (`ARFHoeffdingTreeClassifier` parameter)
        Number of instances a leaf should observe between split attempts.
    split_criterion
        (`ARFHoeffdingTreeClassifier` parameter)
        Split criterion to use.
        - 'gini' - Gini
        - 'info_gain' - Information Gain
    split_confidence
        (`ARFHoeffdingTreeClassifier` parameter)
        Allowed error in split decision, a value closer to 0 takes longer to decide.
    tie_threshold
        (`ARFHoeffdingTreeClassifier` parameter)
        Threshold below which a split will be forced to break ties.
    binary_split
        (`ARFHoeffdingTreeClassifier` parameter)
        If True, only allow binary splits.
    stop_mem_management
        (`ARFHoeffdingTreeClassifier` parameter)
        If True, stop growing as soon as memory limit is hit.
    remove_poor_atts
        (`ARFHoeffdingTreeClassifier` parameter)
        If True, disable poor attributes.
    merit_preprune
        (`ARFHoeffdingTreeClassifier` parameter)
        If True, disable pre-pruning.
    leaf_prediction
        (`ARFHoeffdingTreeClassifier` parameter)
        Prediction mechanism used at leafs.
        - 'mc' - Majority Class
        - 'nb' - Naive Bayes
        - 'nba' - Naive Bayes Adaptive
    nb_threshold
        (`ARFHoeffdingTreeClassifier` parameter)
        Number of instances a leaf should observe before allowing Naive Bayes.
    nominal_attributes
        (`ARFHoeffdingTreeClassifier` parameter)
        List of Nominal attributes. If empty, then assume that all attributes are numerical.
    max_depth
        (`ARFHoeffdingTreeClassifier` parameter)
        The maximum depth a tree can reach. If `None`, the tree will grow indefinitely.
    seed
        If `int`, `seed` is used to seed the random number generator;
        If `RandomState`, `seed` is the random number generator;
        If `None`, the random number generator is the `RandomState` instance
        used by `np.random`.

    Examples
    --------
    >>> from river import synth
    >>> from river import drift
    >>> from river import ensemble
    >>> from river import evaluate
    >>> from river import metrics

    >>> dataset = synth.ConceptDriftStream(seed=42, position=500,
    ...                                    width=40).take(1000)

    >>> model = ensemble.AdaptiveRandomForestClassifier(
    ...     n_models=3,
    ...     seed=42,
    ...     drift_detector=drift.ADWIN(delta=0.15),
    ...     warning_detector=drift.ADWIN(delta=0.2)
    ... )

    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    Accuracy: 64.06%

    References
    ----------
    [^1] Heitor Murilo Gomes, Albert Bifet, Jesse Read, Jean Paul Barddal,
         Fabricio Enembreck, Bernhard Pfharinger, Geoff Holmes, Talel Abdessalem.
         Adaptive random forests for evolving data stream classification.
         In Machine Learning, DOI: 10.1007/s10994-017-5642-8, Springer, 2017.

    """

    _FEATURES_SQRT = "sqrt"
    _FEATURES_LOG2 = "log2"

    def __init__(self,
                 n_models: int = 10,
                 max_features: typing.Union[bool, str, int] = 'sqrt',
                 disable_weighted_vote=False,
                 lambda_value: int = 6,
                 metric: MultiClassMetric = Accuracy(),
                 drift_detector: typing.Union[base.DriftDetector, None] = ADWIN(delta=0.001),
                 warning_detector: typing.Union[base.DriftDetector, None] = ADWIN(delta=0.01),
                 max_size: int = 32,
                 memory_estimate_period: int = 2000000,
                 grace_period: int = 50,
                 split_criterion: str = 'info_gain',
                 split_confidence: float = 0.01,
                 tie_threshold: float = 0.05,
                 binary_split=False,
                 stop_mem_management=False,
                 remove_poor_attrs=False,
                 merit_preprune=True,
                 leaf_prediction: str = 'nba',
                 nb_threshold: int = 0,
                 nominal_attributes: list = None,
                 max_depth: int = None,
                 seed=None):
        super().__init__([None])  # List of models is properly initialized later
        self.models = []
        self.n_models = n_models
        self.max_features = max_features
        self.disable_weighted_vote = disable_weighted_vote
        self.lambda_value = lambda_value
        self.drift_detector = drift_detector
        self.warning_detector = warning_detector
        self.seed = seed
        self._rng = check_random_state(self.seed)   # Actual random number generator
        self.metric = metric

        self._n_samples_seen = 0

        # Adaptive Random Forest Hoeffding Tree configuration
        self.max_size = max_size
        self. memory_estimate_period = memory_estimate_period
        self.grace_period = grace_period
        self.split_criterion = split_criterion
        self.split_confidence = split_confidence
        self.tie_threshold = tie_threshold
        self.binary_split = binary_split
        self.stop_mem_management = stop_mem_management
        self.remove_poor_attrs = remove_poor_attrs
        self.merit_preprune = merit_preprune
        self.leaf_prediction = leaf_prediction
        self.nb_threshold = nb_threshold
        self.nominal_attributes = nominal_attributes
        self.max_depth = max_depth

    def _multiclass(self):
        return True

    def learn_one(self, x: dict, y: base.typing.ClfTarget, **kwargs):
        self._n_samples_seen += 1

        if not self.models:
            self._init_ensemble(list(x.keys()))

        for model in self.models:
            # Get prediction for instance
            y_pred = model.predict_one(x)

            # Update performance evaluator
            model.metric.update(y_true=y, y_pred=y_pred)

            k = self._rng.poisson(lam=self.lambda_value)
            if k > 0:
                # print(self._n_samples_seen)
                model.learn_one(x=x, y=y, sample_weight=k, n_samples_seen=self._n_samples_seen)

        return self

    def predict_proba_one(self, x):

        y_pred = collections.Counter()

        if not self.models:
            self._init_ensemble(features=list(x.keys()))
            return y_pred

        for model in self.models:
            y_proba_temp = model.predict_proba_one(x)
            metric_value = model.metric.get()
            if not self.disable_weighted_vote and metric_value > 0.:
                y_proba_temp = {k: val * metric_value for k, val in y_proba_temp.items()}
            y_pred.update(y_proba_temp)

        total = sum(y_pred.values())
        if total > 0:
            return {label: proba / total for label, proba in y_pred.items()}
        return y_pred

    def reset(self):
        """Reset ARF."""
        self.models = []
        self._n_samples_seen = 0
        self._rng = check_random_state(self.seed)

    def _init_ensemble(self, features: list):
        self._set_max_features(len(features))

        self.models = [
            BaseARFLearner(
                index_original=i,
                base_model=ARFHoeffdingTreeClassifier(
                    max_size=self.max_size,
                    memory_estimate_period=self.memory_estimate_period,
                    grace_period=self.grace_period,
                    split_criterion=self.split_criterion,
                    split_confidence=self.split_confidence,
                    tie_threshold=self.tie_threshold,
                    binary_split=self.binary_split,
                    stop_mem_management=self.stop_mem_management,
                    remove_poor_attrs=self.remove_poor_attrs,
                    merit_preprune=self.merit_preprune,
                    leaf_prediction=self.leaf_prediction,
                    nb_threshold=self.nb_threshold,
                    nominal_attributes=self.nominal_attributes,
                    max_features=self.max_features,
                    max_depth=self.max_depth,
                    seed=self.seed
                ),
                created_on=self._n_samples_seen,
                base_drift_detector=self.drift_detector,
                base_warning_detector=self.warning_detector,
                is_background_learner=False,
                metric=copy.deepcopy(self.metric))
            for i in range(self.n_models)
        ]

    def _set_max_features(self, n_features):
        if self.max_features == 'sqrt':
            self.max_features = round(math.sqrt(n_features))
        elif self.max_features == 'log2':
            self.max_features = round(math.log2(n_features))
        elif isinstance(self.max_features, int):
            # Consider 'max_features' features at each split.
            pass
        elif isinstance(self.max_features, float):
            # Consider 'max_features' as a percentage
            self.max_features = int(self.max_features * n_features)
        elif self.max_features is None:
            self.max_features = n_features
        else:
            raise AttributeError(f"Invalid max_features: {self.max_features}.\n"
                                 f"Valid options are: int [2, M], float (0., 1.],"
                                 f" {self._FEATURES_SQRT}, {self._FEATURES_LOG2}"
                                 )
        # Sanity checks
        # max_features is negative, use max_features + n
        if self.max_features < 0:
            self.max_features += n_features
        # max_features <= 0
        # (m can be negative if max_features is negative and abs(max_features) > n),
        # use max_features = 1
        if self.max_features <= 0:
            self.max_features = 1
        # max_features > n, then use n
        if self.max_features > n_features:
            self.max_features = n_features


class BaseARFLearner(base.Classifier):
    """Base learner class.

    This wrapper class represents a tree member of the forest. It includes a
    base tree model, the background learner, drift detectors and performance
    tracking parameters.

    The main purpose of this class is to train the foreground model.
    Optionally, it monitors drift detection. Depending on the configuration,
    if drift is detected then the foreground model is reset or replaced by a
    background model.

    Parameters
    ----------
    index_original
        Tree index within the ensemble.
    base_model: ARFHoeffdingTreeClassifier
        Tree classifier.
    created_on
        Number of instances seen by the tree.
    base_drift_detector: DriftDetector
        Drift Detection method.
    base_warning_detector
        Warning Detection method.
    is_background_learner
        True if the tree is a background learner.

    """
    def __init__(self,
                 index_original: int,
                 base_model: ARFHoeffdingTreeClassifier,
                 created_on: int,
                 base_drift_detector: base.DriftDetector,
                 base_warning_detector: base.DriftDetector,
                 is_background_learner,
                 metric: MultiClassMetric):
        self.index_original = index_original
        self.base_model = base_model
        self.model = copy.deepcopy(base_model)
        self.created_on = created_on
        self.is_background_learner = is_background_learner
        self.metric = metric
        # Make sure that the metric is not initialized, e.g. when creating background learners.
        self.metric.cm.reset()

        self.background_learner = None

        # Drift and warning detection
        self.base_drift_detector = base_drift_detector  # Drift detector prototype
        self.base_warning_detector = base_warning_detector  # Warning detector prototype

        self.last_drift_on = 0
        self.last_warning_on = 0
        self.n_drifts_detected = 0
        self.n_warnings_detected = 0

        # Initialize drift and warning detectors
        # TODO Replace deepcopy with clone
        if base_drift_detector is not None:
            self._use_drift_detector = True
            self.drift_detector = copy.deepcopy(base_drift_detector)  # Actual detector used
        else:
            self._use_drift_detector = False
            self.drift_detector = None

        if base_warning_detector is not None:
            self._use_background_learner = True
            self.warning_detector = copy.deepcopy(base_warning_detector)  # Actual detector used
        else:
            self._use_background_learner = False
            self.warning_detector = None

    def reset(self, n_samples_seen):
        if self._use_background_learner and self.background_learner is not None:
            # Replace foreground model with background model
            self.model = self.background_learner.model
            self.warning_detector = self.background_learner.warning_detector
            self.drift_detector = self.background_learner.drift_detector
            self.metric = self.background_learner.metric
            self.metric.cm.reset()
            self.created_on = self.background_learner.created_on
            self.background_learner = None
        else:
            # Reset model
            self.model = copy.deepcopy(self.base_model)
            self.metric.cm.reset()
            self.created_on = n_samples_seen
            self.drift_detector = copy.deepcopy(self.base_drift_detector)

    def learn_one(self, x: dict, y: base.typing.ClfTarget, *, sample_weight: int,   # noqa
                  n_samples_seen: int):

        self.model.learn_one(x, y, sample_weight=sample_weight)

        if self.background_learner:
            # Train the background learner
            self.background_learner.model.learn_one(x=x, y=y, sample_weight=sample_weight)

        if self._use_drift_detector and not self.is_background_learner:
            correctly_classifies = self.model.predict_one(x) == y
            # Check for warning only if use_background_learner is active
            if self._use_background_learner:
                self.warning_detector.update(int(not correctly_classifies))
                # Check if there was a change
                if self.warning_detector.change_detected:
                    self.last_warning_on = n_samples_seen
                    self.n_warnings_detected += 1
                    # Create a new background learner object
                    self.background_learner = BaseARFLearner(
                        index_original=self.index_original,
                        base_model=self.model.new_instance(),
                        created_on=n_samples_seen,
                        base_drift_detector=self.base_drift_detector,
                        base_warning_detector=self.base_warning_detector,
                        is_background_learner=True,
                        metric=copy.deepcopy(self.metric)
                    )
                    # Reset the warning detection object for the current object
                    self.warning_detector = copy.deepcopy(self.base_warning_detector)

            # Update the drift detection
            self.drift_detector.update(int(not correctly_classifies))

            # Check if there was a change
            if self.drift_detector.change_detected:
                self.last_drift_on = n_samples_seen
                self.n_drifts_detected += 1
                self.reset(n_samples_seen)

    def predict_one(self, x):
        return self.model.predict_one(x)

    def predict_proba_one(self, x):
        return self.model.predict_proba_one(x)
