from __future__ import annotations

import abc
import collections
import copy
import math
import random
import typing

import numpy as np

from river import base, metrics, stats
from river.drift import ADWIN, NoDrift
from river.tree.hoeffding_tree_classifier import HoeffdingTreeClassifier
from river.tree.hoeffding_tree_regressor import HoeffdingTreeRegressor
from river.tree.nodes.arf_htc_nodes import (
    RandomLeafMajorityClass,
    RandomLeafNaiveBayes,
    RandomLeafNaiveBayesAdaptive,
)
from river.tree.nodes.arf_htr_nodes import RandomLeafAdaptive, RandomLeafMean, RandomLeafModel
from river.tree.splitter import Splitter
from river.utils.random import poisson


class BaseForest(base.Ensemble):
    _FEATURES_SQRT = "sqrt"
    _FEATURES_LOG2 = "log2"

    def __init__(
        self,
        n_models: int,
        max_features: bool | str | int,
        lambda_value: int,
        drift_detector: base.DriftDetector,
        warning_detector: base.DriftDetector,
        metric: metrics.base.MultiClassMetric | metrics.base.RegressionMetric,
        disable_weighted_vote,
        seed,
    ):
        super().__init__([])  # type: ignore
        self.n_models = n_models
        self.max_features = max_features
        self.lambda_value = lambda_value
        self.metric = metric
        self.disable_weighted_vote = disable_weighted_vote
        self.drift_detector = drift_detector
        self.warning_detector = warning_detector
        self.seed = seed

        self._rng = random.Random(self.seed)

        self._warning_detectors: list[base.DriftDetector]
        self._warning_detection_disabled = True
        if not isinstance(self.warning_detector, NoDrift):
            self._warning_detectors = [self.warning_detector.clone() for _ in range(self.n_models)]
            self._warning_detection_disabled = False

        self._drift_detectors: list[base.DriftDetector]
        self._drift_detection_disabled = True
        if not isinstance(self.drift_detector, NoDrift):
            self._drift_detectors = [self.drift_detector.clone() for _ in range(self.n_models)]
            self._drift_detection_disabled = False

        # The background models
        self._background: list[BaseTreeClassifier | BaseTreeRegressor | None] = (
            None if self._warning_detection_disabled else [None] * self.n_models  # type: ignore
        )

        # Performance metrics used for weighted voting/aggregation
        self._metrics = [self.metric.clone() for _ in range(self.n_models)]

        # Drift and warning logging
        self._warning_tracker: dict = (
            collections.defaultdict(int) if not self._warning_detection_disabled else None  # type: ignore
        )
        self._drift_tracker: dict = (
            collections.defaultdict(int) if not self._drift_detection_disabled else None  # type: ignore
        )

    @property
    def _min_number_of_models(self):
        return 0

    @classmethod
    def _unit_test_params(cls):
        yield {"n_models": 3}

    def _unit_test_skips(self):
        return {"check_shuffle_features_no_impact"}

    @abc.abstractmethod
    def _drift_detector_input(
        self,
        tree_id: int,
        y_true,
        y_pred,
    ) -> int | float:
        raise NotImplementedError

    @abc.abstractmethod
    def _new_base_model(self) -> BaseTreeClassifier | BaseTreeRegressor:
        raise NotImplementedError

    def n_warnings_detected(self, tree_id: int | None = None) -> int:
        """Get the total number of concept drift warnings detected, or the number on an individual
        tree basis (optionally).

        Parameters
        ----------
        tree_id
            The number of the base learner in the ensemble: `[0, self.n_models - 1]. If `None`,
            the total number of warnings is returned instead.

        Returns
        -------
            The number of concept drift warnings detected.

        """

        if self._warning_detection_disabled:
            return 0

        if tree_id is None:
            return sum(self._warning_tracker.values())

        return self._warning_tracker[tree_id]

    def n_drifts_detected(self, tree_id: int | None = None) -> int:
        """Get the total number of concept drifts detected, or such number on an individual
        tree basis (optionally).

        Parameters
        ----------
        tree_id
            The number of the base learner in the ensemble: `[0, self.n_models - 1]. If `None`,
            the total number of warnings is returned instead.

        Returns
        -------
            The number of concept drifts detected.

        """

        if self._drift_detection_disabled:
            return 0

        if tree_id is None:
            return sum(self._drift_tracker.values())

        return self._drift_tracker[tree_id]

    def learn_one(self, x: dict, y: base.typing.Target, **kwargs):
        if len(self) == 0:
            self._init_ensemble(sorted(x.keys()))

        for i, model in enumerate(self):
            y_pred = model.predict_one(x)

            # Update performance evaluator
            self._metrics[i].update(
                y_true=y,
                y_pred=(
                    model.predict_proba_one(x)
                    if isinstance(self.metric, metrics.base.ClassificationMetric)
                    and not self.metric.requires_labels
                    else y_pred
                ),
            )

            k = poisson(rate=self.lambda_value, rng=self._rng)
            if k > 0:
                if not self._warning_detection_disabled and self._background[i] is not None:
                    self._background[i].learn_one(x=x, y=y, w=k)  # type: ignore

                model.learn_one(x=x, y=y, w=k)

                drift_input = None
                if not self._warning_detection_disabled:
                    drift_input = self._drift_detector_input(i, y, y_pred)
                    self._warning_detectors[i].update(drift_input)

                    if self._warning_detectors[i].drift_detected:
                        self._background[i] = self._new_base_model()  # type: ignore
                        # Reset the warning detector for the current object
                        self._warning_detectors[i] = self.warning_detector.clone()

                        # Update warning tracker
                        self._warning_tracker[i] += 1

                if not self._drift_detection_disabled:
                    drift_input = (
                        drift_input
                        if drift_input is not None
                        else self._drift_detector_input(i, y, y_pred)
                    )
                    self._drift_detectors[i].update(drift_input)

                    if self._drift_detectors[i].drift_detected:
                        if not self._warning_detection_disabled and self._background[i] is not None:
                            self.data[i] = self._background[i]
                            self._background[i] = None
                            self._warning_detectors[i] = self.warning_detector.clone()
                            self._drift_detectors[i] = self.drift_detector.clone()
                            self._metrics[i] = self.metric.clone()
                        else:
                            self.data[i] = self._new_base_model()
                            self._drift_detectors[i] = self.drift_detector.clone()
                            self._metrics[i] = self.metric.clone()

                        # Update warning tracker
                        self._drift_tracker[i] += 1

    def _init_ensemble(self, features: list):
        self._set_max_features(len(features))
        self.data = [self._new_base_model() for _ in range(self.n_models)]

    def _set_max_features(self, n_features):
        if self.max_features == "sqrt":
            self.max_features = round(math.sqrt(n_features))
        elif self.max_features == "log2":
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
            raise AttributeError(
                f"Invalid max_features: {self.max_features}.\n"
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


class BaseTreeClassifier(HoeffdingTreeClassifier):
    """Adaptive Random Forest Hoeffding Tree Classifier.

    This is the base-estimator of the Adaptive Random Forest classifier.
    This variant of the Hoeffding Tree classifier includes the `max_features`
    parameter, which defines the number of randomly selected features to be
    considered at each split.

    """

    def __init__(
        self,
        max_features: int = 2,
        grace_period: int = 200,
        max_depth: int | None = None,
        split_criterion: str = "info_gain",
        delta: float = 1e-7,
        tau: float = 0.05,
        leaf_prediction: str = "nba",
        nb_threshold: int = 0,
        nominal_attributes: list | None = None,
        splitter: Splitter | None = None,
        binary_split: bool = False,
        min_branch_fraction: float = 0.01,
        max_share_to_split: float = 0.99,
        max_size: float = 100.0,
        memory_estimate_period: int = 1000000,
        stop_mem_management: bool = False,
        remove_poor_attrs: bool = False,
        merit_preprune: bool = True,
        rng: random.Random | None = None,
    ):
        super().__init__(
            grace_period=grace_period,
            max_depth=max_depth,
            split_criterion=split_criterion,
            delta=delta,
            tau=tau,
            leaf_prediction=leaf_prediction,
            nb_threshold=nb_threshold,
            nominal_attributes=nominal_attributes,
            splitter=splitter,
            binary_split=binary_split,
            min_branch_fraction=min_branch_fraction,
            max_share_to_split=max_share_to_split,
            max_size=max_size,
            memory_estimate_period=memory_estimate_period,
            stop_mem_management=stop_mem_management,
            remove_poor_attrs=remove_poor_attrs,
            merit_preprune=merit_preprune,
        )

        self.max_features = max_features
        self.rng = rng

    def _new_leaf(self, initial_stats=None, parent=None):
        if initial_stats is None:
            initial_stats = {}

        if parent is None:
            depth = 0
        else:
            depth = parent.depth + 1

        if self._leaf_prediction == self._MAJORITY_CLASS:
            return RandomLeafMajorityClass(
                initial_stats,
                depth,
                self.splitter,
                self.max_features,
                self.rng,
            )
        elif self._leaf_prediction == self._NAIVE_BAYES:
            return RandomLeafNaiveBayes(
                initial_stats,
                depth,
                self.splitter,
                self.max_features,
                self.rng,
            )
        else:  # NAIVE BAYES ADAPTIVE (default)
            return RandomLeafNaiveBayesAdaptive(
                initial_stats,
                depth,
                self.splitter,
                self.max_features,
                self.rng,
            )


class BaseTreeRegressor(HoeffdingTreeRegressor):
    """ARF Hoeffding Tree regressor.

    This is the base-estimator of the Adaptive Random Forest regressor.
    This variant of the Hoeffding Tree regressor includes the `max_features`
    parameter, which defines the number of randomly selected features to be
    considered at each split.

    """

    def __init__(
        self,
        max_features: int = 2,
        grace_period: int = 200,
        max_depth: int | None = None,
        delta: float = 1e-7,
        tau: float = 0.05,
        leaf_prediction: str = "adaptive",
        leaf_model: base.Regressor | None = None,
        model_selector_decay: float = 0.95,
        nominal_attributes: list | None = None,
        splitter: Splitter | None = None,
        min_samples_split: int = 5,
        binary_split: bool = False,
        max_size: float = 100.0,
        memory_estimate_period: int = 1000000,
        stop_mem_management: bool = False,
        remove_poor_attrs: bool = False,
        merit_preprune: bool = True,
        rng: random.Random | None = None,
    ):
        super().__init__(
            grace_period=grace_period,
            max_depth=max_depth,
            delta=delta,
            tau=tau,
            leaf_prediction=leaf_prediction,
            leaf_model=leaf_model,
            model_selector_decay=model_selector_decay,
            nominal_attributes=nominal_attributes,
            splitter=splitter,
            min_samples_split=min_samples_split,
            binary_split=binary_split,
            max_size=max_size,
            memory_estimate_period=memory_estimate_period,
            stop_mem_management=stop_mem_management,
            remove_poor_attrs=remove_poor_attrs,
            merit_preprune=merit_preprune,
        )

        self.max_features = max_features
        self.rng = rng

    def _new_leaf(self, initial_stats=None, parent=None):  # noqa
        """Create a new learning node.

        The type of learning node depends on the tree configuration.
        """

        if parent is not None:
            depth = parent.depth + 1
        else:
            depth = 0

        leaf_model = None
        if self.leaf_prediction in {self._MODEL, self._ADAPTIVE}:
            if parent is None:
                leaf_model = copy.deepcopy(self.leaf_model)
            else:
                try:
                    leaf_model = copy.deepcopy(parent._leaf_model)  # noqa
                except AttributeError:
                    leaf_model = copy.deepcopy(self.leaf_model)

        if self.leaf_prediction == self._TARGET_MEAN:
            return RandomLeafMean(
                initial_stats,
                depth,
                self.splitter,
                self.max_features,
                self.rng,
            )
        elif self.leaf_prediction == self._MODEL:
            return RandomLeafModel(
                initial_stats,
                depth,
                self.splitter,
                self.max_features,
                self.rng,
                leaf_model=leaf_model,
            )
        else:  # adaptive learning node
            new_adaptive = RandomLeafAdaptive(
                initial_stats,
                depth,
                self.splitter,
                self.max_features,
                self.rng,
                leaf_model=leaf_model,
            )
            if parent is not None and isinstance(parent, RandomLeafAdaptive):
                new_adaptive._fmse_mean = parent._fmse_mean  # noqa
                new_adaptive._fmse_model = parent._fmse_model  # noqa

            return new_adaptive


class ARFClassifier(BaseForest, base.Classifier):
    """Adaptive Random Forest classifier.

    The 3 most important aspects of Adaptive Random Forest [^1] are:

    1. inducing diversity through re-sampling

    2. inducing diversity through randomly selecting subsets of features for
       node splits

    3. drift detectors per base tree, which cause selective resets in response
       to drifts

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
    lambda_value
        The lambda value for bagging (lambda=6 corresponds to Leveraging Bagging).
    metric
        Metric used to track trees performance within the ensemble.
        Defaults to `metrics.Accuracy()`.
    disable_weighted_vote
        If `True`, disables the weighted vote prediction.
    drift_detector
        Drift Detection method. Set to None to disable Drift detection.
        Defaults to `drift.ADWIN(delta=0.001)`.
    warning_detector
        Warning Detection method. Set to None to disable warning detection.
        Defaults to `drift.ADWIN(delta=0.01)`.
    grace_period
        [*Tree parameter*] Number of instances a leaf should observe between
        split attempts.
    max_depth
        [*Tree parameter*] The maximum depth a tree can reach. If `None`, the
        tree will grow until the system recursion limit.
    split_criterion
        [*Tree parameter*] Split criterion to use.<br/>
        - 'gini' - Gini<br/>
        - 'info_gain' - Information Gain<br/>
        - 'hellinger' - Hellinger Distance
    delta
        [*Tree parameter*] Allowed error in split decision, a value closer to 0
        takes longer to decide.
    tau
        [*Tree parameter*] Threshold below which a split will be forced to break
        ties.
    leaf_prediction
        [*Tree parameter*] Prediction mechanism used at leafs.<br/>
        - 'mc' - Majority Class<br/>
        - 'nb' - Naive Bayes<br/>
        - 'nba' - Naive Bayes Adaptive
    nb_threshold
        [*Tree parameter*] Number of instances a leaf should observe before
        allowing Naive Bayes.
    nominal_attributes
        [*Tree parameter*] List of Nominal attributes. If empty, then assume that
        all attributes are numerical.
    splitter
        [*Tree parameter*] The Splitter or Attribute Observer (AO) used to monitor the class
        statistics of numeric features and perform splits. Splitters are available in the
        `tree.splitter` module. Different splitters are available for classification and
        regression tasks. Classification and regression splitters can be distinguished by their
        property `is_target_class`. This is an advanced option. Special care must be taken when
        choosing different splitters. By default, `tree.splitter.GaussianSplitter` is used
        if `splitter` is `None`.
    binary_split
        [*Tree parameter*] If True, only allow binary splits.
    min_branch_fraction
        [*Tree parameter*] The minimum percentage of observed data required for branches
        resulting from split candidates. To validate a split candidate, at least two resulting
        branches must have a percentage of samples greater than `min_branch_fraction`. This
        criterion prevents unnecessary splits when the majority of instances are concentrated
        in a single branch.
    max_share_to_split
        [*Tree parameter*] Only perform a split in a leaf if the proportion of elements
        in the majority class is smaller than this parameter value. This parameter avoids
        performing splits when most of the data belongs to a single class.
    max_size
        [*Tree parameter*] Maximum memory (MiB) consumed by the tree.
    memory_estimate_period
        [*Tree parameter*] Number of instances between memory consumption checks.
    stop_mem_management
        [*Tree parameter*] If True, stop growing as soon as memory limit is hit.
    remove_poor_attrs
        [*Tree parameter*] If True, disable poor attributes to reduce memory usage.
    merit_preprune
        [*Tree parameter*] If True, enable merit-based tree pre-pruning.
    seed
        Random seed for reproducibility.

    Examples
    --------

    >>> from river import evaluate
    >>> from river import forest
    >>> from river import metrics
    >>> from river.datasets import synth

    >>> dataset = synth.ConceptDriftStream(
    ...     seed=42,
    ...     position=500,
    ...     width=40
    ... ).take(1000)

    >>> model = forest.ARFClassifier(seed=8, leaf_prediction="mc")

    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    Accuracy: 67.97%

    The total number of warnings and drifts detected, respectively
    >>> model.n_warnings_detected(), model.n_drifts_detected()
    (2, 1)

    The number of warnings detected by tree number 2
    >>> model.n_warnings_detected(2)
    1

    And the corresponding number of actual concept drift detected
    >>> model.n_drifts_detected(2)
    1

    References
    ----------
    [^1]: Heitor Murilo Gomes, Albert Bifet, Jesse Read, Jean Paul Barddal,
         Fabricio Enembreck, Bernhard Pfharinger, Geoff Holmes, Talel Abdessalem.
         Adaptive random forests for evolving data stream classification.
         In Machine Learning, DOI: 10.1007/s10994-017-5642-8, Springer, 2017.

    """

    def __init__(
        self,
        n_models: int = 10,
        max_features: bool | str | int = "sqrt",
        lambda_value: int = 6,
        metric: metrics.base.MultiClassMetric | None = None,
        disable_weighted_vote=False,
        drift_detector: base.DriftDetector | None = None,
        warning_detector: base.DriftDetector | None = None,
        # Tree parameters
        grace_period: int = 50,
        max_depth: int | None = None,
        split_criterion: str = "info_gain",
        delta: float = 0.01,
        tau: float = 0.05,
        leaf_prediction: str = "nba",
        nb_threshold: int = 0,
        nominal_attributes: list | None = None,
        splitter: Splitter | None = None,
        binary_split: bool = False,
        min_branch_fraction: float = 0.01,
        max_share_to_split: float = 0.99,
        max_size: float = 100.0,
        memory_estimate_period: int = 2_000_000,
        stop_mem_management: bool = False,
        remove_poor_attrs: bool = False,
        merit_preprune: bool = True,
        seed: int | None = None,
    ):
        super().__init__(
            n_models=n_models,
            max_features=max_features,
            lambda_value=lambda_value,
            metric=metric or metrics.Accuracy(),
            disable_weighted_vote=disable_weighted_vote,
            drift_detector=drift_detector or ADWIN(delta=0.001),
            warning_detector=warning_detector or ADWIN(delta=0.01),
            seed=seed,
        )

        # Tree parameters
        self.grace_period = grace_period
        self.max_depth = max_depth
        self.split_criterion = split_criterion
        self.delta = delta
        self.tau = tau
        self.leaf_prediction = leaf_prediction
        self.nb_threshold = nb_threshold
        self.nominal_attributes = nominal_attributes
        self.splitter = splitter
        self.binary_split = binary_split
        self.min_branch_fraction = min_branch_fraction
        self.max_share_to_split = max_share_to_split
        self.max_size = max_size
        self.memory_estimate_period = memory_estimate_period
        self.stop_mem_management = stop_mem_management
        self.remove_poor_attrs = remove_poor_attrs
        self.merit_preprune = merit_preprune

    @property
    def _mutable_attributes(self):
        return {
            "max_features",
            "lambda_value",
            "grace_period",
            "delta",
            "tau",
        }

    @property
    def _multiclass(self):
        return True

    def predict_proba_one(self, x: dict) -> dict[base.typing.ClfTarget, float]:
        y_pred: typing.Counter = collections.Counter()

        if len(self) == 0:
            self._init_ensemble(sorted(x.keys()))
            return y_pred  # type: ignore

        for i, model in enumerate(self):
            y_proba_temp = model.predict_proba_one(x)
            metric_value = self._metrics[i].get()
            if not self.disable_weighted_vote and metric_value > 0.0:
                y_proba_temp = {k: val * metric_value for k, val in y_proba_temp.items()}
            y_pred.update(y_proba_temp)

        total = sum(y_pred.values())
        if total > 0:
            return {label: proba / total for label, proba in y_pred.items()}
        return y_pred  # type: ignore

    def _new_base_model(self):
        return BaseTreeClassifier(
            max_features=self.max_features,
            grace_period=self.grace_period,
            split_criterion=self.split_criterion,
            delta=self.delta,
            tau=self.tau,
            leaf_prediction=self.leaf_prediction,
            nb_threshold=self.nb_threshold,
            nominal_attributes=self.nominal_attributes,
            splitter=self.splitter,
            max_depth=self.max_depth,
            binary_split=self.binary_split,
            min_branch_fraction=self.min_branch_fraction,
            max_share_to_split=self.max_share_to_split,
            max_size=self.max_size,
            memory_estimate_period=self.memory_estimate_period,
            stop_mem_management=self.stop_mem_management,
            remove_poor_attrs=self.remove_poor_attrs,
            merit_preprune=self.merit_preprune,
            rng=self._rng,
        )

    def _drift_detector_input(
        self, tree_id: int, y_true: base.typing.ClfTarget, y_pred: base.typing.ClfTarget
    ) -> int | float:
        return int(not y_true == y_pred)


class ARFRegressor(BaseForest, base.Regressor):
    """Adaptive Random Forest regressor.

    The 3 most important aspects of Adaptive Random Forest [^1] are:

    1. inducing diversity through re-sampling

    2. inducing diversity through randomly selecting subsets of features for
       node splits

    3. drift detectors per base tree, which cause selective resets in response
       to drifts

    Notice that this implementation is slightly different from the original
    algorithm proposed in [^2]. The `HoeffdingTreeRegressor` is used as base
    learner, instead of `FIMT-DD`. It also adds a new strategy to monitor the
    predictions and check for concept drifts. The deviations of the predictions
    to the target are monitored and normalized in the [0, 1] range to fulfill ADWIN's
    requirements. We assume that the data subjected to the normalization follows
    a normal distribution, and thus, lies within the interval of the mean $\\pm3\\sigma$.

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
    lambda_value
        The lambda value for bagging (lambda=6 corresponds to Leveraging Bagging).
    metric
        Metric used to track trees performance within the ensemble. Depending,
        on the configuration, this metric is also used to weight predictions
        from the members of the ensemble. Defaults to `metrics.MSE()`.
    aggregation_method
        The method to use to aggregate predictions in the ensemble.<br/>
        - 'mean'<br/>
        - 'median' - If selected will disable the weighted vote.
    disable_weighted_vote
        If `True`, disables the weighted vote prediction, i.e. does not assign
        weights to individual tree's predictions and uses the arithmetic mean
        instead. Otherwise will use the `metric` value to weight predictions.
    drift_detector
        Drift Detection method. Set to None to disable Drift detection.
        Defaults to `drift.ADWIN(0.001)`.
    warning_detector
        Warning Detection method. Set to None to disable warning detection.
        Defaults to `drift.ADWIN(0.01)`.
    grace_period
        [*Tree parameter*] Number of instances a leaf should observe between
        split attempts.
    max_depth
        [*Tree parameter*] The maximum depth a tree can reach. If `None`, the
        tree will grow until the system recursion limit.
    delta
        [*Tree parameter*] Allowed error in split decision, a value closer to 0
        takes longer to decide.
    tau
        [*Tree parameter*] Threshold below which a split will be forced to break
        ties.
    leaf_prediction
        [*Tree parameter*] Prediction mechanism used at leaves.</br>
        - 'mean' - Target mean</br>
        - 'model' - Uses the model defined in `leaf_model`</br>
        - 'adaptive' - Chooses between 'mean' and 'model' dynamically</br>
    leaf_model
        [*Tree parameter*] The regression model used to provide responses if
        `leaf_prediction='model'`. If not provided, an instance of
        `river.linear_model.LinearRegression` with the default hyperparameters
         is used.
    model_selector_decay
        [*Tree parameter*] The exponential decaying factor applied to the learning models'
        squared errors, that are monitored if `leaf_prediction='adaptive'`. Must be
        between `0` and `1`. The closer to `1`, the more importance is going to
        be given to past observations. On the other hand, if its value
        approaches `0`, the recent observed errors are going to have more
        influence on the final decision.
    nominal_attributes
        [*Tree parameter*] List of Nominal attributes. If empty, then assume that
        all attributes are numerical.
    splitter
        [*Tree parameter*] The Splitter or Attribute Observer (AO) used to monitor the class
        statistics of numeric features and perform splits. Splitters are available in the
        `tree.splitter` module. Different splitters are available for classification and
        regression tasks. Classification and regression splitters can be distinguished by their
        property `is_target_class`. This is an advanced option. Special care must be taken when
        choosing different splitters.By default, `tree.splitter.EBSTSplitter` is used if
        `splitter` is `None`.
    min_samples_split
        [*Tree parameter*] The minimum number of samples every branch resulting from a split
        candidate must have to be considered valid.
    binary_split
        [*Tree parameter*] If True, only allow binary splits.
    max_size
        [*Tree parameter*] Maximum memory (MiB) consumed by the tree.
    memory_estimate_period
        [*Tree parameter*] Number of instances between memory consumption checks.
    stop_mem_management
        [*Tree parameter*] If True, stop growing as soon as memory limit is hit.
    remove_poor_attrs
        [*Tree parameter*] If True, disable poor attributes to reduce memory usage.
    merit_preprune
        [*Tree parameter*] If True, enable merit-based tree pre-pruning.
    seed
        Random seed for reproducibility.

    References
    ----------
    [^1]: Gomes, H.M., Bifet, A., Read, J., Barddal, J.P., Enembreck, F.,
          Pfharinger, B., Holmes, G. and Abdessalem, T., 2017. Adaptive random
          forests for evolving data stream classification. Machine Learning,
          106(9-10), pp.1469-1495.

    [^2]: Gomes, H.M., Barddal, J.P., Boiko, L.E., Bifet, A., 2018.
          Adaptive random forests for data stream regression. ESANN 2018.

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import forest
    >>> from river import metrics
    >>> from river import preprocessing

    >>> dataset = datasets.TrumpApproval()

    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     forest.ARFRegressor(seed=42)
    ... )

    >>> metric = metrics.MAE()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    MAE: 0.772113

    """

    _MEAN = "mean"
    _MEDIAN = "median"
    _VALID_AGGREGATION_METHOD = [_MEAN, _MEDIAN]

    def __init__(
        self,
        # Forest parameters
        n_models: int = 10,
        max_features="sqrt",
        aggregation_method: str = "median",
        lambda_value: int = 6,
        metric: metrics.base.RegressionMetric | None = None,
        disable_weighted_vote=True,
        drift_detector: base.DriftDetector | None = None,
        warning_detector: base.DriftDetector | None = None,
        # Tree parameters
        grace_period: int = 50,
        max_depth: int | None = None,
        delta: float = 0.01,
        tau: float = 0.05,
        leaf_prediction: str = "adaptive",
        leaf_model: base.Regressor | None = None,
        model_selector_decay: float = 0.95,
        nominal_attributes: list | None = None,
        splitter: Splitter | None = None,
        min_samples_split: int = 5,
        binary_split: bool = False,
        max_size: float = 500.0,
        memory_estimate_period: int = 2_000_000,
        stop_mem_management: bool = False,
        remove_poor_attrs: bool = False,
        merit_preprune: bool = True,
        seed: int | None = None,
    ):
        super().__init__(
            n_models=n_models,
            max_features=max_features,
            lambda_value=lambda_value,
            metric=metric or metrics.MSE(),
            disable_weighted_vote=disable_weighted_vote,
            drift_detector=drift_detector or ADWIN(0.001),
            warning_detector=warning_detector or ADWIN(0.01),
            seed=seed,
        )

        # Tree parameters
        self.grace_period = grace_period
        self.max_depth = max_depth
        self.delta = delta
        self.tau = tau
        self.leaf_prediction = leaf_prediction
        self.leaf_model = leaf_model
        self.model_selector_decay = model_selector_decay
        self.nominal_attributes = nominal_attributes
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.binary_split = binary_split
        self.max_size = max_size
        self.memory_estimate_period = memory_estimate_period
        self.stop_mem_management = stop_mem_management
        self.remove_poor_attrs = remove_poor_attrs
        self.merit_preprune = merit_preprune

        if aggregation_method in self._VALID_AGGREGATION_METHOD:
            self.aggregation_method = aggregation_method
        else:
            raise ValueError(
                f"Invalid aggregation_method: {aggregation_method}.\n"
                f"Valid values are: {self._VALID_AGGREGATION_METHOD}"
            )

        # Used to normalize the input for the drift trackers
        self._drift_norm = [stats.Var() for _ in range(self.n_models)]

    @property
    def _mutable_attributes(self):
        return {
            "max_features",
            "aggregation_method",
            "lambda_value",
            "grace_period",
            "delta",
            "tau",
            "model_selector_decay",
        }

    def predict_one(self, x: dict) -> base.typing.RegTarget:
        if len(self) == 0:
            self._init_ensemble(sorted(x.keys()))
            return 0.0  # type: ignore

        y_pred = np.zeros(self.n_models)

        if not self.disable_weighted_vote and self.aggregation_method != self._MEDIAN:
            weights = np.zeros(self.n_models)
            sum_weights = 0.0
            for i, model in enumerate(self):
                y_pred[i] = model.predict_one(x)
                weights[i] = self._metrics[i].get()
                sum_weights += weights[i]

            if sum_weights != 0:
                # The higher the error, the worse is the tree
                weights = sum_weights - weights
                # Normalize weights to sum up to 1
                weights /= weights.sum()
                y_pred *= weights
        else:
            for i, model in enumerate(self):
                y_pred[i] = model.predict_one(x)

        if self.aggregation_method == self._MEAN:
            y_pred = y_pred.mean()
        else:
            y_pred = float(np.median(y_pred))  # type: ignore

        return float(y_pred)  # type: ignore

    def _new_base_model(self):
        return BaseTreeRegressor(
            max_features=self.max_features,
            grace_period=self.grace_period,
            max_depth=self.max_depth,
            delta=self.delta,
            tau=self.tau,
            leaf_prediction=self.leaf_prediction,
            leaf_model=self.leaf_model,
            model_selector_decay=self.model_selector_decay,
            nominal_attributes=self.nominal_attributes,
            splitter=self.splitter,
            binary_split=self.binary_split,
            max_size=self.max_size,
            memory_estimate_period=self.memory_estimate_period,
            stop_mem_management=self.stop_mem_management,
            remove_poor_attrs=self.remove_poor_attrs,
            merit_preprune=self.merit_preprune,
            rng=self._rng,
        )

    def _drift_detector_input(
        self,
        tree_id: int,
        y_true: int | float,
        y_pred: int | float,
    ) -> int | float:
        drift_input = y_true - y_pred
        self._drift_norm[tree_id].update(drift_input)

        if self._drift_norm[tree_id].mean.n == 1:
            return 0.5  # The expected error is the normalized mean error

        sd = math.sqrt(self._drift_norm[tree_id].get())

        # We assume the error follows a normal distribution -> (empirical rule)
        # 99.73% of the values lie  between [mean - 3*sd, mean + 3*sd]. We
        # assume this range for the normalized data. Hence, we can apply the
        # min-max norm to cope with  ADWIN's requirements
        return (drift_input + 3 * sd) / (6 * sd) if sd > 0 else 0.5

    @property
    def valid_aggregation_method(self):
        """Valid aggregation_method values."""
        return self._VALID_AGGREGATION_METHOD
