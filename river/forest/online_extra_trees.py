from __future__ import annotations

import abc
import collections
import math
import numbers
import random
import sys

from river import base, drift, metrics, tree, utils
from river.tree.nodes.et_nodes import ETLeafAdaptive, ETLeafMean, ETLeafModel
from river.tree.splitter.random_splitter import RegRandomSplitter


class Sampler(base.Base, abc.ABC):
    @abc.abstractmethod
    def __call__(self, rate, rng) -> int:
        pass


class ConstantSampler(Sampler):
    def __call__(self, rate, rng):
        return 1


class BaggingSampler(Sampler):
    def __call__(self, rate, rng):
        return utils.random.poisson(rate, rng)


class SubBaggingSampler(Sampler):
    def __call__(self, rate, rng) -> int:
        return 1 if rng.random() <= rate else 0


class ExtraTrees(base.Ensemble, metaclass=abc.ABCMeta):
    _FEATURES_SQRT = "sqrt"
    _FEATURES_LOG2 = "log2"
    _FEATURES_RANDOM = "random"

    _BAGGING = "bagging"
    _SUBBAGGING = "subbagging"

    _DETECTION_ALL = "all"
    _DETECTION_DROP = "drop"
    _DETECTION_OFF = "off"

    def __init__(
        self,
        n_models: int,
        max_features: bool | str | int,
        resampling_strategy: str | None,
        resampling_rate: int | float,
        detection_mode: str,
        warning_detector: base.DriftDetector | None,
        drift_detector: base.DriftDetector | None,
        max_depth: int | None,
        randomize_tree_depth: bool,
        track_metric: metrics.base.MultiClassMetric | metrics.base.RegressionMetric,
        disable_weighted_vote: bool,
        split_buffer_size: int,
        seed: int | None,
    ):
        self.data = []
        self.n_models = n_models
        self.max_features = max_features

        if resampling_strategy not in [None, self._BAGGING, self._SUBBAGGING]:
            raise ValueError(f"Invalid resampling strategy: {resampling_strategy}")
        self.resampling_strategy = resampling_strategy

        self.resampling_rate: int | float = 0
        if self.resampling_strategy is not None:
            if self.resampling_strategy == self._BAGGING:
                if resampling_rate < 1:
                    raise ValueError(
                        "'resampling_rate' must be an integer greater than or"
                        "equal to 1, when resample_strategy='bagging'."
                    )
                # Cast to integer (online bagging using poisson sampling)
                self.resampling_rate = int(resampling_rate)

            if self.resampling_strategy == self._SUBBAGGING:
                if not 0 < resampling_rate <= 1:
                    raise ValueError(
                        "resampling_rate must be a float in the interval (0, 1],"
                        "when resampling_strategy='subbagging'."
                    )
                self.resampling_rate = resampling_rate

        if detection_mode not in [
            self._DETECTION_ALL,
            self._DETECTION_DROP,
            self._DETECTION_OFF,
        ]:
            raise ValueError(
                f"Invalid drift detection mode. Valid values are: '{self._DETECTION_ALL}',"
                f" {self._DETECTION_DROP}, and '{self._DETECTION_OFF}'."
            )

        self.detection_mode = detection_mode
        self.warning_detector = warning_detector or drift.ADWIN(delta=0.01)
        self.drift_detector = drift_detector or drift.ADWIN(delta=0.001)

        self.max_depth = max_depth
        self.randomize_tree_depth = randomize_tree_depth
        self.track_metric = track_metric
        self.disable_weighted_vote = disable_weighted_vote
        self.split_buffer_size = split_buffer_size
        self.seed = seed

        # The predictive performance of each tree
        self._perfs: list = []
        # Keep a running estimate of the sum of performances
        self._perf_sum: float = 0

        # Number of times a tree will use each instance to learn from it
        self._weight_sampler = self.__weight_sampler_factory()

        # General statistics
        # Counter of the number of instances each ensemble member has processed (instance weights
        # are not accounted for, just the number of instances)
        self._sample_counter: collections.Counter = collections.Counter()
        # Total of samples processed by the Extra Trees ensemble
        self._total_instances: float = 0
        # Number of warnings triggered
        self._n_warnings: collections.Counter = collections.Counter()
        # Number of drifts detected
        self._n_drifts: collections.Counter = collections.Counter()
        # Number of tree swaps
        self._n_tree_swaps: collections.Counter = collections.Counter()

        self._background_trees: dict[int, tree.hoeffding_tree.HoeffdingTree] = {}
        # Initialize drift detectors and select the detection mode procedure
        if self.detection_mode == self._DETECTION_ALL:
            self._warn_detectors = {i: self.warning_detector.clone() for i in range(self.n_models)}
            self._drift_detectors = {i: self.drift_detector.clone() for i in range(self.n_models)}
        elif self.detection_mode == self._DETECTION_DROP:
            self._warn_detectors = {}
            self._drift_detectors = {i: self.drift_detector.clone() for i in range(self.n_models)}
        else:  # detection_mode: "off"
            self._warn_detectors = {}
            self._drift_detectors = {}
        self._detect = self.__detection_mode_factory()

        # Set the rng
        self._rng = random.Random(seed)

    @abc.abstractmethod
    def _new_member(self, max_features, max_depth, seed) -> base.Classifier | base.Regressor:
        pass

    @abc.abstractmethod
    def _drift_input(self, y, y_hat) -> int | float:
        pass

    def _calculate_tree_depth(self) -> int | float:
        max_depth = self.max_depth or math.inf
        if not self.randomize_tree_depth:
            return max_depth
        return self._rng.randint(1, max_depth if not math.isinf(max_depth) else 9999)  # type: ignore

    def _calculate_max_features(self, n_features) -> int:
        if self.max_features == self._FEATURES_RANDOM:
            # Generate a random integer
            return self._rng.randint(2, n_features)
        else:
            if self.max_features == self._FEATURES_SQRT:
                max_feat = round(math.sqrt(n_features))
            elif self.max_features == self._FEATURES_LOG2:
                max_feat = round(math.log2(n_features))
            elif isinstance(self.max_features, int):
                max_feat = n_features
            elif isinstance(self.max_features, float):
                # Consider 'max_features' as a percentage
                max_feat = int(self.max_features * n_features)
            elif self.max_features is None:
                max_feat = n_features
            else:
                raise AttributeError(
                    f"Invalid max_features: {self.max_features}.\n"
                    f"Valid options are: int [2, M], float (0., 1.],"
                    f" {self._FEATURES_SQRT}, {self._FEATURES_LOG2}"
                )

            # Sanity checks
            # max_feat is negative, use max_feat + n
            if max_feat < 0:
                max_feat += n_features
            # max_feat <= 0
            # (m can be negative if max_feat is negative and abs(max_feat) > n),
            # use max_features = 1
            if max_feat <= 0:
                max_feat = 1
            # max_feat > n, then use n
            if max_feat > n_features:
                max_feat = n_features

            return max_feat

    def _init_trees(self, n_features: int):
        for _ in range(self.n_models):
            self.data.append(
                self._new_member(
                    max_features=self._calculate_max_features(n_features),
                    max_depth=self._calculate_tree_depth(),
                    seed=self._rng.randint(0, sys.maxsize),  # randomly creates a new seed
                )
            )
            self._perfs.append(self.track_metric.clone())

    def __weight_sampler_factory(self) -> Sampler:
        if self.resampling_strategy == self._BAGGING:
            return BaggingSampler()
        elif self.resampling_strategy == self._SUBBAGGING:
            return SubBaggingSampler()
        else:
            return ConstantSampler()

    @staticmethod
    def _detection_mode_all(
        drift_detector: base.DriftDetector,
        warning_detector: base.DriftDetector,
        detector_input: numbers.Number,
    ) -> tuple[bool, bool]:
        in_warning = warning_detector.update(detector_input).drift_detected
        in_drift = drift_detector.update(detector_input).drift_detected

        return in_drift, in_warning

    @staticmethod
    def _detection_mode_drop(
        drift_detector: base.DriftDetector,
        warning_detector: base.DriftDetector,
        detector_input: numbers.Number,
    ) -> tuple[bool, bool]:
        in_drift = drift_detector.update(detector_input).drift_detected

        return in_drift, False

    @staticmethod
    def _detection_mode_off(
        drift_detector: base.DriftDetector,
        warning_detector: base.DriftDetector,
        detector_input: numbers.Number,
    ) -> tuple[bool, bool]:
        return False, False

    def __detection_mode_factory(self):
        if self.detection_mode == self._DETECTION_ALL:
            return self._detection_mode_all
        elif self.detection_mode == self._DETECTION_DROP:
            return self._detection_mode_drop
        else:
            return self._detection_mode_off

    def learn_one(self, x, y):
        if not self.models:
            self._init_trees(len(x))

        self._total_instances += 1
        trained = []
        for i, model in enumerate(self.models):
            y_hat = model.predict_one(x)
            in_drift, in_warning = self._detect(
                self._drift_detectors.get(i),
                self._warn_detectors.get(i),
                self._drift_input(y, y_hat),
            )

            if in_warning:
                self._background_trees[i] = self._new_member(
                    max_features=self._calculate_max_features(len(x)),
                    max_depth=self._calculate_tree_depth(),
                    seed=self._rng.randint(0, sys.maxsize),  # randomly creates a new seed
                )
                # Reset the warning detector
                self._warn_detectors[i] = self.warning_detector.clone()
                # Update statistics
                self._n_warnings.update([i])

            # Drift detected: time to change (swap or reset) the affected tree
            if in_drift:
                if i in self._background_trees:
                    self.data[i] = self._background_trees[i]
                    del self._background_trees[i]

                    self._n_tree_swaps.update([i])
                else:
                    self.data[i] = self._new_member(
                        max_features=self._calculate_max_features(len(x)),
                        max_depth=self._calculate_tree_depth(),
                        seed=self._rng.randint(0, sys.maxsize),  # randomly creates a new seed
                    )
                # Reset the drift detector
                self._drift_detectors[i] = self.drift_detector.clone()
                # Update statistics
                self._n_drifts.update([i])
                # Also reset tree's error estimates
                self._perf_sum -= self._perfs[i].get()
                self._perfs[i] = self.track_metric.clone()
                self._perf_sum += self._perfs[i].get()
                # And the number of observations of the new model
                self._sample_counter[i] = 0

            # Remove the old performance estimate
            self._perf_sum -= self._perfs[i].get()
            # Update metric
            self._perfs[i].update(y, y_hat)
            # Add the new performance estimate
            self._perf_sum += self._perfs[i].get()

            # Define the weight of the instance
            w = self._weight_sampler(self.resampling_rate, self._rng)
            if w == 0:  # Skip model update if w is zero
                continue

            model.learn_one(x, y, sample_weight=w)

            if i in self._background_trees:
                self._background_trees[i].learn_one(x, y, sample_weight=w)

            trained.append(i)

        # Increase by one the count of instances observed by each trained model
        self._sample_counter.update(trained)

        return self

    # Properties
    @property
    def n_warnings(self) -> collections.Counter:
        """The number of warnings detected per ensemble member."""
        return self._n_warnings

    @property
    def n_drifts(self) -> collections.Counter:
        """The number of concept drifts detected per ensemble member."""
        return self._n_drifts

    @property
    def n_tree_swaps(self) -> collections.Counter:
        """The number of performed alternate tree swaps.

        Not applicable if the warning detectors are disabled.
        """
        return self._n_tree_swaps

    @property
    def total_instances(self) -> float:
        """The total number of instances processed by the ensemble."""
        return self._total_instances

    @property
    def instances_per_tree(self) -> collections.Counter:
        """The number of instances processed by each one of the current forest members.

        Each time a concept drift is detected, the count corresponding to the affected tree is
        reset.
        """
        return self._sample_counter

    @classmethod
    def _unit_test_params(cls):
        yield {"n_models": 3}

    def _unit_test_skips(self):
        return {"check_shuffle_features_no_impact"}


class ETRegressor(tree.HoeffdingTreeRegressor):
    """Extra Tree regressor.

    This is the base-estimator of the Extra Trees regressor.
    This variant of the Hoeffding Tree regressor includes the `max_features` parameter,
    which defines the number of randomly selected features to be considered at each split.
    It also evaluates split candidates randomly.

    """

    def __init__(
        self,
        max_features,
        grace_period,
        max_depth,
        delta,
        tau,
        leaf_prediction,
        leaf_model,
        model_selector_decay,
        nominal_attributes,
        min_samples_split,
        binary_split,
        max_size,
        memory_estimate_period,
        stop_mem_management,
        remove_poor_attrs,
        merit_preprune,
        split_buffer_size,
        seed,
    ):
        self.max_features = max_features
        self.split_buffer_size = split_buffer_size
        self.seed = seed
        self._rng = random.Random(self.seed)

        super().__init__(
            grace_period=grace_period,
            max_depth=max_depth,
            delta=delta,
            tau=tau,
            leaf_prediction=leaf_prediction,
            leaf_model=leaf_model,
            model_selector_decay=model_selector_decay,
            nominal_attributes=nominal_attributes,
            splitter=RegRandomSplitter(
                seed=self._rng.randint(0, sys.maxsize),
                buffer_size=self.split_buffer_size,
            ),
            min_samples_split=min_samples_split,
            binary_split=binary_split,
            max_size=max_size,
            memory_estimate_period=memory_estimate_period,
            stop_mem_management=stop_mem_management,
            remove_poor_attrs=remove_poor_attrs,
            merit_preprune=merit_preprune,
        )

    def _new_learning_node(self, initial_stats=None, parent=None):  # noqa
        """Create a new learning node.
        The type of learning node depends on the tree configuration.
        """

        if parent is not None:
            depth = parent.depth + 1
        else:
            depth = 0

        # Generate a random seed for the new learning node
        seed = self._rng.randint(0, sys.maxsize)

        leaf_model = None
        if self.leaf_prediction in {self._MODEL, self._ADAPTIVE}:
            if parent is None:
                leaf_model = self.leaf_model.clone()
            else:
                leaf_model = parent._leaf_model.clone(include_attributes=True)  # noqa

        if self.leaf_prediction == self._TARGET_MEAN:
            return ETLeafMean(
                initial_stats,
                depth,
                self.splitter,
                self.max_features,
                seed,
            )
        elif self.leaf_prediction == self._MODEL:
            return ETLeafModel(
                initial_stats,
                depth,
                self.splitter,
                self.max_features,
                seed,
                leaf_model=leaf_model,
            )
        else:  # adaptive learning node
            new_adaptive = ETLeafAdaptive(
                initial_stats,
                depth,
                self.splitter,
                self.max_features,
                seed,
                leaf_model=leaf_model,
            )
            if parent is not None and isinstance(parent, ETLeafAdaptive):
                new_adaptive._fmse_mean = parent._fmse_mean  # noqa
                new_adaptive._fmse_model = parent._fmse_model  # noqa

            return new_adaptive


class OXTRegressor(ExtraTrees, base.Regressor):
    """Online Extra Trees regressor.

    The online Extra Trees[^1] ensemble takes some steps further into randomization when
    compared to Adaptive Random Forests (ARF). A subspace of the feature space is considered
    at each split attempt, as ARF does, and online bagging or subbagging can also be
    (optionally) used. Nonetheless, Extra Trees randomizes the split candidates evaluated by each
    leaf node (just a single split is tested by numerical feature, which brings significant
    speedups to the ensemble), and might also randomize the maximum depth of the forest members,
    as well as the size of the feature subspace processed by each of its trees' leaves.

    On the other hand, OXT suffers from a cold-start problem. As the splits are random, the
    predictive performance in small samples is usually worse than using a deterministic
    split approach, such as the one used by ARF.

    Parameters
    ----------
    n_models
        The number of trees in the ensemble.
    max_features
        Max number of attributes for each node split.</br>
        - If int, then consider `max_features` at each split.</br>
        - If float, then `max_features` is a percentage and `int(max_features * n_features)`
        features are considered per split.</br>
        - If "sqrt", then `max_features=sqrt(n_features)`.</br>
        - If "log2", then `max_features=log2(n_features)`.</br>
        - If "random", then `max_features` will assume a different random number in the interval
        `[2, n_features]` for each tree leaf.</br>
        - If None, then `max_features=n_features`.
    resampling_strategy
        The chosen instance resampling strategy:</br>
        - If `None`, no resampling will be done and the trees will process all instances.
        - If `'baggging'`, online bagging will be performed (sampling with replacement).
        - If `'subbagging'`, online subbagging will be performed (sampling without replacement).
    resampling_rate
        Only valid if `resampling_strategy` is not None. Controls the parameters of the resampling
        strategy.</br>.
        - If `resampling_strategy='bagging'`, must be an integer greater than or equal to 1 that
        parameterizes the poisson distribution used to simulate bagging in online learning
        settings. It acts as the lambda parameter of Oza Bagging and Leveraging Bagging.</br>
        - If `resampling_strategy='subbagging'`, must be a float in the interval $(0, 1]$ that
        controls the chance of each instance being used by a tree for learning.
    detection_mode
        The concept drift detection mode in which the forest operates. Valid values are:</br>
        - "all": creates both warning and concept drift detectors. If a warning is detected,
        an alternate tree starts being trained in the background. If the warning trigger escalates
        to a concept drift, the affected tree is replaced by the alternate tree.</br>
        - "drop": only the concept drift detectors are created. If a drift is detected, the
        affected tree is dropped and replaced by a new tree.</br>
        - "off": disables the concept drift adaptation capabilities. The forest will act as if
        the processed stream is stationary.
    warning_detector
        The detector that will be used to trigger concept drift warnings. Defaults to
        `drift.ADWIN(0.01)`.
    drift_detector
        The detector used to detect concept drifts. Defaults to `drift.ADWIN(0.001)`.
    max_depth
        The maximum depth the ensemble members might reach. If `None`, the trees will grow
        indefinitely.
    randomize_tree_depth
        Whether or not randomize the maximum depth of each tree in the ensemble. If `max_depth`
        is provided, it is going to act as an upper bound to generate the maximum depth for each
        tree.
    track_metric
        The performance metric used to weight predictions. Defaults to `metrics.MAE()`.
    disable_weighted_vote
        Defines whether or not to use predictions weighted by each trees' prediction performance.
    split_buffer_size
        Defines the size of the buffer used by the tree splitters when determining the feature
        range and a random split point in this interval.
    seed
        Random seed to support reproducibility.
    grace_period
        [*Tree parameter*] Number of instances a leaf should observe between
        split attempts.
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
    min_samples_split
        [*Tree parameter*] The minimum number of samples every branch resulting from a split
        candidate must have to be considered valid.
    binary_split
        [*Tree parameter*] If True, only allow binary splits.
    max_size
        [*Tree parameter*] Maximum memory (MB) consumed by the tree.
    memory_estimate_period
        [*Tree parameter*] Number of instances between memory consumption checks.
    stop_mem_management
        [*Tree parameter*] If True, stop growing as soon as memory limit is hit.
    remove_poor_attrs
        [*Tree parameter*] If True, disable poor attributes to reduce memory usage.
    merit_preprune
        [*Tree parameter*] If True, enable merit-based tree pre-pruning.

    Notes
    -----
    As the Online Extra Trees change the way in which Hoeffding Trees perform split attempts
    and monitor numerical input features, some of the parameters of the vanilla Hoeffding Tree
    algorithms are not available.

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import forest

    >>> dataset = datasets.synth.Friedman(seed=42).take(5000)

    >>> model = forest.OXTRegressor(n_models=3, seed=42)

    >>> metric = metrics.RMSE()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    RMSE: 3.127311

    References
    ----------
    [^1]: Mastelini, S. M., Nakano, F. K., Vens, C., & de Leon Ferreira, A. C. P. (2022).
    Online Extra Trees Regressor. IEEE Transactions on Neural Networks and Learning Systems.

    """

    def __init__(
        self,
        n_models: int = 10,
        max_features: bool | str | int = "sqrt",
        resampling_strategy: str | None = "subbagging",
        resampling_rate: int | float = 0.5,
        detection_mode: str = "all",
        warning_detector: base.DriftDetector | None = None,
        drift_detector: base.DriftDetector | None = None,
        max_depth: int | None = None,
        randomize_tree_depth: bool = False,
        track_metric: metrics.base.RegressionMetric | None = None,
        disable_weighted_vote: bool = True,
        split_buffer_size: int = 5,
        seed: int | None = None,
        grace_period: int = 50,
        delta: float = 0.01,
        tau: float = 0.05,
        leaf_prediction: str = "adaptive",
        leaf_model: base.Regressor | None = None,
        model_selector_decay: float = 0.95,
        nominal_attributes: list | None = None,
        min_samples_split: int = 5,
        binary_split: bool = False,
        max_size: int = 500,
        memory_estimate_period: int = 2_000_000,
        stop_mem_management: bool = False,
        remove_poor_attrs: bool = False,
        merit_preprune: bool = True,
    ):
        super().__init__(
            n_models=n_models,
            max_features=max_features,
            resampling_strategy=resampling_strategy,
            resampling_rate=resampling_rate,
            detection_mode=detection_mode,
            warning_detector=warning_detector,
            drift_detector=drift_detector,
            max_depth=max_depth,
            randomize_tree_depth=randomize_tree_depth,
            track_metric=track_metric or metrics.MAE(),
            disable_weighted_vote=disable_weighted_vote,
            split_buffer_size=split_buffer_size,
            seed=seed,
        )

        # Tree parameters
        self.grace_period = grace_period
        self.delta = delta
        self.tau = tau
        self.leaf_prediction = leaf_prediction
        self.leaf_model = leaf_model
        self.model_selector_decay = model_selector_decay
        self.nominal_attributes = nominal_attributes
        self.min_samples_split = min_samples_split
        self.binary_split = binary_split
        self.max_size = max_size
        self.memory_estimate_period = memory_estimate_period
        self.stop_mem_management = stop_mem_management
        self.remove_poor_attrs = remove_poor_attrs
        self.merit_preprune = merit_preprune

    def _new_member(self, max_features, max_depth, seed):
        return ETRegressor(
            max_features=max_features,
            grace_period=self.grace_period,
            max_depth=max_depth,
            delta=self.delta,
            tau=self.tau,
            leaf_prediction=self.leaf_prediction,
            leaf_model=self.leaf_model,
            model_selector_decay=self.model_selector_decay,
            nominal_attributes=self.nominal_attributes,
            min_samples_split=self.min_samples_split,
            binary_split=self.binary_split,
            max_size=self.max_size,
            memory_estimate_period=self.memory_estimate_period,
            stop_mem_management=self.stop_mem_management,
            remove_poor_attrs=self.remove_poor_attrs,
            merit_preprune=self.merit_preprune,
            split_buffer_size=self.split_buffer_size,
            seed=seed,
        )

    def _drift_input(self, y, y_hat) -> int | float:
        return abs(y - y_hat)

    def predict_one(self, x: dict) -> base.typing.RegTarget:
        if not self.models:
            self._init_trees(len(x))
            return 0.0  # type: ignore

        if not self.disable_weighted_vote:
            preds = []
            weights = []

            for perf, model in zip(self._perfs, self.models):
                preds.append(model.predict_one(x))
                weights.append(perf.get())

            sum_weights = sum(weights)
            if sum_weights != 0:
                if self.track_metric.bigger_is_better:
                    preds = [(w / sum_weights) * pred for w, pred in zip(weights, preds)]
                else:
                    weights = [(1 + 1e-8) / (w + 1e-8) for w in weights]
                    sum_weights = sum(weights)
                    preds = [(w / sum_weights) * pred for w, pred in zip(weights, preds)]
                return sum(preds)
        else:
            preds = [model.predict_one(x) for model in self.models]

        return sum(preds) / len(preds)
