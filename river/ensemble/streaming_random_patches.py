from __future__ import annotations

import collections
import itertools
import math
import random

import numpy as np

from river import base
from river.drift import ADWIN
from river.metrics import MAE, Accuracy
from river.metrics.base import ClassificationMetric, Metric, RegressionMetric
from river.tree import HoeffdingTreeClassifier, HoeffdingTreeRegressor
from river.utils.random import poisson


class BaseSRPEnsemble(base.Wrapper, base.Ensemble):
    """Base class for the sRP ensemble family"""

    _TRAIN_RANDOM_SUBSPACES = "subspaces"
    _TRAIN_RESAMPLING = "resampling"
    _TRAIN_RANDOM_PATCHES = "patches"

    _FEATURES_SQRT = "sqrt"
    _FEATURES_SQRT_INV = "rmsqrt"

    _VALID_TRAINING_METHODS = {
        _TRAIN_RANDOM_PATCHES,
        _TRAIN_RESAMPLING,
        _TRAIN_RESAMPLING,
    }

    def __init__(
        self,
        model: base.Estimator | None = None,
        n_models: int = 100,
        subspace_size: int | float | str = 0.6,
        training_method: str = "patches",
        lam: float = 6.0,
        drift_detector: base.DriftDetector | None = None,
        warning_detector: base.DriftDetector | None = None,
        disable_detector: str = "off",
        disable_weighted_vote: bool = False,
        seed: int | None = None,
        metric: Metric | None = None,
    ):
        # List of models is properly initialized later
        super().__init__([])  # type: ignore
        self.model = model  # Not restricted to a specific base estimator.
        self.n_models = n_models
        self.subspace_size = subspace_size
        self.training_method = training_method
        self.lam = lam
        self.drift_detector = drift_detector
        self.warning_detector = warning_detector
        self.disable_weighted_vote = disable_weighted_vote
        self.disable_detector = disable_detector
        self.metric = metric
        self.seed = seed
        self._rng = random.Random(self.seed)

        self._n_samples_seen = 0
        self._subspaces: list = []

        # defined by extended classes
        self._base_learner_class: BaseSRPClassifier | BaseSRPRegressor | None = None

    @property
    def _min_number_of_models(self):
        return 0

    @property
    def _wrapped_model(self):
        return self.model

    @classmethod
    def _unit_test_params(cls):
        yield {"n_models": 3, "seed": 42}

    def _unit_test_skips(self):  # noqa
        return {
            "check_shuffle_features_no_impact",
            "check_emerging_features",
            "check_disappearing_features",
        }

    def learn_one(self, x: dict, y: base.typing.Target, **kwargs):
        self._n_samples_seen += 1

        if not self:
            self._init_ensemble(list(x.keys()))

        for model in self:
            # Get prediction for instance
            y_pred = model.predict_one(x)

            # Update performance evaluator
            if y_pred is not None:
                model.metric.update(y_true=y, y_pred=y_pred)

            # Train using random subspaces without resampling,
            # i.e. all instances are used for training.
            if self.training_method == self._TRAIN_RANDOM_SUBSPACES:
                k = 1
            # Train using random patches or resampling,
            # thus we simulate online bagging with Poisson(lambda=...)
            else:
                k = poisson(rate=self.lam, rng=self._rng)
                if k == 0:
                    continue
            model.learn_one(x=x, y=y, sample_weight=k, n_samples_seen=self._n_samples_seen)

        return self

    def _generate_subspaces(self, features: list):
        n_features = len(features)

        self._subspaces = [None] * self.n_models

        if self.training_method != self._TRAIN_RESAMPLING:
            # Set subspaces - This only applies to subspaces and random patches options

            # 1. Calculate the number of features k
            if isinstance(self.subspace_size, float) and 0.0 < self.subspace_size <= 1:
                k = self.subspace_size
                percent = (1.0 + k) / 1.0 if k < 0 else k
                k = round(n_features * percent)
                if k < 2:
                    k = round(n_features * percent) + 1
            elif isinstance(self.subspace_size, int) and self.subspace_size > 2:
                # k is a fixed number of features
                k = self.subspace_size
            elif self.subspace_size == self._FEATURES_SQRT:
                # k is sqrt(M)+1
                k = round(math.sqrt(n_features)) + 1
            elif self.subspace_size == self._FEATURES_SQRT_INV:
                # k is M-(sqrt(M)+1)
                k = n_features - round(math.sqrt(n_features)) + 1
            else:
                raise ValueError(
                    f"Invalid subspace_size: {self.subspace_size}.\n"
                    f"Valid options are: int [2, M], float (0., 1.],"
                    f" {self._FEATURES_SQRT}, {self._FEATURES_SQRT_INV}"
                )
            if k < 0:
                # k is negative, calculate M - k
                k = n_features + k

            # 2. Generate subspaces. The subspaces is a 2D array of shape
            # (n_estimators, k) where each row contains the k-feature indices
            # to be used by each estimator.
            if k != 0 and k < n_features:
                # For low dimensionality it is better to avoid more than
                # 1 classifier with the same subspace, thus we generate all
                # possible combinations of subsets of features and select
                # without replacement.
                # n_features is the total number of features and k is the
                # actual size of a subspace.
                if n_features <= 20 or k < 2:
                    if k == 1 and n_features > 2:
                        k = 2
                    # Generate n_models subspaces from all possible
                    # feature combinations of size k
                    self._subspaces = []
                    for i, combination in enumerate(
                        itertools.cycle(itertools.combinations(features, k))
                    ):
                        if i == self.n_models:
                            break
                        self._subspaces.append(list(combination))
                # For high dimensionality we can't generate all combinations
                # as it is too expensive (memory). On top of that, the chance
                # of repeating a subspace is lower, so we randomly generate
                # subspaces without worrying about repetitions.
                else:
                    self._subspaces = [
                        random_subspace(all_features=features, k=k, rng=self._rng)
                        for _ in range(self.n_models)
                    ]
            else:
                # k == 0 or k > n_features (subspace size is larger than the
                # number of features), then default to re-sampling
                self.training_method = self._TRAIN_RESAMPLING

    def _init_ensemble(self, features: list):
        self._generate_subspaces(features=features)
        subspace_indexes = list(
            range(self.n_models)
        )  # For matching subspaces with ensemble members
        if (
            self.training_method == self._TRAIN_RANDOM_PATCHES
            or self.training_method == self._TRAIN_RANDOM_SUBSPACES
        ):
            # Shuffle indexes
            self._rng.shuffle(subspace_indexes)

        # Initialize the ensemble
        for i in range(self.n_models):
            # If self.training_method == self._TRAIN_RESAMPLING then subspace is None
            subspace = self._subspaces[subspace_indexes[i]]
            self.append(
                self._base_learner_class(  # type: ignore
                    idx_original=i,
                    model=self.model,
                    metric=self.metric,
                    created_on=self._n_samples_seen,
                    drift_detector=self.drift_detector,
                    warning_detector=self.warning_detector,
                    is_background_learner=False,
                    rng=self._rng,
                    features=subspace,
                )
            )

    def reset(self):
        self.data = []
        self._n_samples_seen = 0
        self._rng = random.Random(self.seed)


class BaseSRPEstimator:
    """Base class for estimators (classifiers or regressors) in SRP"""

    def __init__(
        self,
        idx_original: int,
        model: base.Estimator,
        metric: Metric,
        created_on: int,
        drift_detector: base.DriftDetector,
        warning_detector: base.DriftDetector,
        is_background_learner,
        rng: random.Random,
        features=None,
    ):
        self.idx_original = idx_original
        self.created_on = created_on
        self.model = model.clone()
        self.metric = metric.clone()

        # Store current model subspace representation of the original instances
        self.features = features

        # Drift and warning detection
        if drift_detector is not None:
            self.disable_drift_detector = False
            self.drift_detector = drift_detector.clone()  # Actual detector used
        else:
            self.disable_drift_detector = True
            self.drift_detector = None

        if warning_detector is not None:
            self.disable_background_learner = False
            self.warning_detector = warning_detector.clone()  # Actual detector used
        else:
            self.disable_background_learner = True
            self.warning_detector = None

        # Background learner
        self.is_background_learner = is_background_learner

        # Statistics
        self.n_drifts_detected = 0
        self.n_warnings_detected = 0

        # Random number generator (initialized)
        self.rng = rng

        # Background learner
        self._background_learner: BaseSRPClassifier | BaseSRPRegressor | None = None

    def _trigger_warning(self, all_features, n_samples_seen: int):
        # Randomly generate a new subspace from all the original features
        subspace = (
            None
            if self.features is None
            else random_subspace(all_features=all_features, k=len(self.features), rng=self.rng)
        )

        # Initialize the background learner
        self._background_learner = self.__class__(  # type: ignore
            idx_original=self.idx_original,
            model=self.model,
            metric=self.metric,
            created_on=n_samples_seen,
            drift_detector=self.drift_detector,
            warning_detector=self.warning_detector,
            is_background_learner=True,
            rng=self.rng,
            features=subspace,
        )
        # Hard-reset the warning method
        self.warning_detector = self.warning_detector.clone()

    def reset(self, all_features: list, n_samples_seen: int):
        if not self.disable_background_learner and self._background_learner is not None:
            # Replace model with the corresponding background model
            self.model = self._background_learner.model
            self.drift_detector = self._background_learner.drift_detector
            self.warning_detector = self._background_learner.warning_detector
            self.metric = self._background_learner.metric
            self.created_on = self._background_learner.created_on
            self.features = self._background_learner.features
            self._background_learner = None
        else:
            # Randomly generate a new subspace from all the original features
            subspace = (
                None
                if self.features is None
                else random_subspace(all_features=all_features, k=len(self.features), rng=self.rng)
            )
            # Reset model
            self.model = self.model.clone()
            self.metric = self.metric.clone()
            self.created_on = n_samples_seen
            self.drift_detector = self.drift_detector.clone()
            self.features = subspace


def random_subspace(all_features: list, k: int, rng: random.Random):
    """Utility function to generate a random feature subspace of length k

    Parameters
    ----------
    all_features
        List of possible features to select from.
    k
        Subspace length.
    rng
        Random number generator (initialized).
    """
    return rng.sample(all_features, k=k)


class SRPClassifier(BaseSRPEnsemble, base.Classifier):
    """Streaming Random Patches ensemble classifier.

    The Streaming Random Patches (SRP) [^1] is an ensemble method that
    simulates bagging or random subspaces. The default algorithm uses both
    bagging and random subspaces, namely Random Patches. The default base
    estimator is a Hoeffding Tree, but other base estimators can be used
    (differently from random forest variations).

    Parameters
    ----------
    model
        The base estimator.
    n_models
        Number of members in the ensemble.
    subspace_size
        Number of features per subset for each classifier where `M` is the
        total number of features.<br/>
        A negative value means `M - subspace_size`.<br/>
        Only applies when using random subspaces or random patches.<br/>
        * If `int` indicates the number of features to use. Valid range [2, M]. <br/>
        * If `float` indicates the percentage of features to use, Valid range (0., 1.]. <br/>
        * 'sqrt' - `sqrt(M)+1`<br/>
        * 'rmsqrt' - Residual from `M-(sqrt(M)+1)`
    training_method
        The training method to use.<br/>
        * 'subspaces' - Random subspaces.<br/>
        * 'resampling' - Resampling.<br/>
        * 'patches' - Random patches.
    lam
        Lambda value for resampling.
    drift_detector
        Drift detector.
    warning_detector
        Warning detector.
    disable_detector
        Option to disable drift detectors:<br/>
        * If `'off'`, detectors are enabled.<br/>
        * If `'drift'`, disables concept drift detection and the background learner.<br/>
        * If `'warning'`, disables the background learner and ensemble members are
         reset if drift is detected.<br/>
    disable_weighted_vote
        If True, disables weighted voting.
    seed
        Random number generator seed for reproducibility.
    metric
        The metric to track members performance within the ensemble. This
        implementation assumes that larger values are better when using
        weighted votes.

    Examples
    --------

    >>> from river import ensemble
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river.datasets import synth
    >>> from river import tree

    >>> dataset = synth.ConceptDriftStream(
    ...     seed=42,
    ...     position=500,
    ...     width=50
    ... ).take(1000)

    >>> base_model = tree.HoeffdingTreeClassifier(
    ...     grace_period=50, delta=0.01,
    ...     nominal_attributes=['age', 'car', 'zipcode']
    ... )
    >>> model = ensemble.SRPClassifier(
    ...     model=base_model, n_models=3, seed=42,
    ... )

    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    Accuracy: 72.77%

    Notes
    -----
    This implementation uses `n_models=10` as default given the impact on
    processing time. The optimal number of models depends on the data and
    resources available.

    References
    ----------
    [^1]: Heitor Murilo Gomes, Jesse Read, Albert Bifet.
          Streaming Random Patches for Evolving Data Stream Classification.
          IEEE International Conference on Data Mining (ICDM), 2019.

    """

    def __init__(
        self,
        model: base.Estimator | None = None,
        n_models: int = 10,
        subspace_size: int | float | str = 0.6,
        training_method: str = "patches",
        lam: int = 6,
        drift_detector: base.DriftDetector | None = None,
        warning_detector: base.DriftDetector | None = None,
        disable_detector: str = "off",
        disable_weighted_vote: bool = False,
        seed: int | None = None,
        metric: ClassificationMetric | None = None,
    ):
        if model is None:
            model = HoeffdingTreeClassifier(grace_period=50, delta=0.01)

        if drift_detector is None:
            drift_detector = ADWIN(delta=1e-5)

        if warning_detector is None:
            warning_detector = ADWIN(delta=1e-4)

        if disable_detector == "off":
            pass
        elif disable_detector == "drift":
            drift_detector = None
            warning_detector = None
        elif disable_detector == "warning":
            warning_detector = None
        else:
            raise AttributeError(
                f"{disable_detector} is not a valid value for disable_detector.\n"
                f"Valid options are: 'off', 'drift', 'warning'"
            )

        if metric is None:
            metric = Accuracy()

        super().__init__(
            model=model,
            n_models=n_models,
            subspace_size=subspace_size,
            training_method=training_method,
            lam=lam,
            drift_detector=drift_detector,
            warning_detector=warning_detector,
            disable_detector=disable_detector,
            disable_weighted_vote=disable_weighted_vote,
            seed=seed,
            metric=metric,
        )

        self._base_learner_class = BaseSRPClassifier  # type: ignore

    def predict_proba_one(self, x, **kwargs):
        y_pred = collections.Counter()

        if not self.models:
            self._init_ensemble(features=list(x.keys()))
            return y_pred

        for model in self.models:
            y_proba_temp = model.predict_proba_one(x, **kwargs)
            metric_value = model.metric.get()
            if not self.disable_weighted_vote and metric_value > 0.0:
                y_proba_temp = {k: val * metric_value for k, val in y_proba_temp.items()}
            y_pred.update(y_proba_temp)

        total = sum(y_pred.values())
        if total > 0:
            return {label: proba / total for label, proba in y_pred.items()}
        return y_pred


class BaseSRPClassifier(BaseSRPEstimator):
    """Class representing the base learner of SRPClassifier."""

    def __init__(
        self,
        idx_original: int,
        model: base.Classifier,
        metric: ClassificationMetric,
        created_on: int,
        drift_detector: base.DriftDetector,
        warning_detector: base.DriftDetector,
        is_background_learner,
        rng: random.Random,
        features=None,
    ):
        super().__init__(
            idx_original=idx_original,
            model=model,
            metric=metric,
            created_on=created_on,
            drift_detector=drift_detector,
            warning_detector=warning_detector,
            is_background_learner=is_background_learner,
            rng=rng,
            features=features,
        )

    def learn_one(
        self,
        x: dict,
        y: base.typing.ClfTarget,
        *,
        sample_weight: int,
        n_samples_seen: int,
        **kwargs,
    ):
        if self.features is not None:
            # Select the subset of features to use
            x_subset = {k: x[k] for k in self.features if k in x}
        else:
            # Use all features
            x_subset = x

        # TODO Find a way to verify if the model natively supports sample_weight
        for _ in range(int(sample_weight)):
            self.model.learn_one(x=x_subset, y=y, **kwargs)

        if self._background_learner:
            # Train the background learner
            # Note: Pass the original instance x so features are correctly
            # selected based on the corresponding subspace
            self._background_learner.learn_one(
                x=x, y=y, sample_weight=sample_weight, n_samples_seen=n_samples_seen  # type: ignore
            )

        if not self.disable_drift_detector and not self.is_background_learner:
            correctly_classifies = self.model.predict_one(x_subset) == y
            # Check for warnings only if the background learner is active
            if not self.disable_background_learner:
                # Update the warning detection method
                self.warning_detector.update(int(not correctly_classifies))
                # Check if there was a change
                if self.warning_detector.drift_detected:
                    all_features = list(x.keys())
                    self.n_warnings_detected += 1
                    self._trigger_warning(all_features=all_features, n_samples_seen=n_samples_seen)

            # ===== Drift detection =====
            # Update the drift detection method
            self.drift_detector.update(int(not correctly_classifies))
            # Check if there was a change
            if self.drift_detector.drift_detected:
                all_features = list(x.keys())
                self.n_drifts_detected += 1
                # There was a change, reset the model
                self.reset(all_features=all_features, n_samples_seen=n_samples_seen)

    def predict_proba_one(self, x, **kwargs):
        # Select the features to use
        x_subset = {k: x[k] for k in self.features if k in x} if self.features is not None else x

        return self.model.predict_proba_one(x_subset, **kwargs)

    def predict_one(self, x: dict, **kwargs) -> base.typing.ClfTarget:
        y_pred = self.predict_proba_one(x, **kwargs)
        if y_pred:
            return max(y_pred, key=y_pred.get)
        return None  # type: ignore


class SRPRegressor(BaseSRPEnsemble, base.Regressor):
    """Streaming Random Patches ensemble regressor.

    The Streaming Random Patches [^1] ensemble method for regression trains
    each base learner on a subset of features and instances from the
    original data, namely a random patch. This strategy to enforce
    diverse base models is similar to the one in the random forest,
    yet it is not restricted to using decision trees as base learner.

    This method is an adaptation of [^2] for regression.

    Parameters
    ----------
    model
        The base estimator.
    n_models
        Number of members in the ensemble.
    subspace_size
        Number of features per subset for each classifier where `M` is the
        total number of features.<br/>
        A negative value means `M - subspace_size`.<br/>
        Only applies when using random subspaces or random patches.<br/>
        * If `int` indicates the number of features to use. Valid range [2, M]. <br/>
        * If `float` indicates the percentage of features to use, Valid range (0., 1.]. <br/>
        * 'sqrt' - `sqrt(M)+1`<br/>
        * 'rmsqrt' - Residual from `M-(sqrt(M)+1)`
    training_method
        The training method to use.<br/>
        * 'subspaces' - Random subspaces.<br/>
        * 'resampling' - Resampling.<br/>
        * 'patches' - Random patches.
    lam
        Lambda value for bagging.
    drift_detector
        Drift detector.
    warning_detector
        Warning detector.
    disable_detector
        Option to disable drift detectors:<br/>
        * If `'off'`, detectors are enabled.<br/>
        * If `'drift'`, disables concept drift detection and the background learner.<br/>
        * If `'warning'`, disables the background learner and ensemble members are
         reset if drift is detected.<br/>
    disable_weighted_vote
        If True, disables weighted voting.
    drift_detection_criteria
        The criteria used to track drifts.<br/>
        * 'error' - absolute error.<br/>
        * 'prediction' - predicted target values.
    aggregation_method
        The method to use to aggregate predictions in the ensemble.<br/>
        * 'mean'<br/>
        * 'median'
    seed
        Random number generator seed for reproducibility.
    metric
        The metric to track members performance within the ensemble.

    Examples
    --------

    >>> from river import ensemble
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river.datasets import synth
    >>> from river import tree

    >>> dataset = synth.FriedmanDrift(
    ...     drift_type='gsg',
    ...     position=(350, 750),
    ...     transition_window=200,
    ...     seed=42
    ... ).take(1000)

    >>> base_model = tree.HoeffdingTreeRegressor(grace_period=50)
    >>> model = ensemble.SRPRegressor(
    ...     model=base_model,
    ...     training_method="patches",
    ...     n_models=3,
    ...     seed=42
    ... )

    >>> metric = metrics.R2()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    R2: 0.571117

    Notes
    -----
    This implementation uses `n_models=10` as default given the impact on
    processing time. The optimal number of models depends on the data and
    resources available.

    References
    ----------
    [^1]: Heitor Gomes, Jacob Montiel, Saulo Martiello Mastelini,
          Bernhard Pfahringer, and Albert Bifet.
          On Ensemble Techniques for Data Stream Regression.
          IJCNN'20. International Joint Conference on Neural Networks. 2020.

    [^2]: Heitor Murilo Gomes, Jesse Read, Albert Bifet.
          Streaming Random Patches for Evolving Data Stream Classification.
          IEEE International Conference on Data Mining (ICDM), 2019.

    """

    _MEAN = "mean"
    _MEDIAN = "median"
    _ERROR = "error"
    _PREDICTION = "prediction"

    def __init__(
        self,
        model: base.Regressor | None = None,
        n_models: int = 10,
        subspace_size: int | float | str = 0.6,
        training_method: str = "patches",
        lam: int = 6,
        drift_detector: base.DriftDetector | None = None,
        warning_detector: base.DriftDetector | None = None,
        disable_detector: str = "off",
        disable_weighted_vote: bool = True,
        drift_detection_criteria: str = "error",
        aggregation_method: str = "mean",
        seed=None,
        metric: RegressionMetric | None = None,
    ):
        # Check arguments for parent class
        if model is None:
            model = HoeffdingTreeRegressor(grace_period=50, delta=0.01)

        if drift_detector is None:
            drift_detector = ADWIN(delta=1e-5)

        if warning_detector is None:
            warning_detector = ADWIN(delta=1e-4)

        if disable_detector == "off":
            pass
        elif disable_detector == "drift":
            drift_detector = None
            warning_detector = None
        elif disable_detector == "warning":
            warning_detector = None
        else:
            raise AttributeError(
                f"{disable_detector} is not a valid value for disable_detector.\n"
                f"Valid options are: 'off', 'drift', 'warning'"
            )

        if metric is None:
            metric = MAE()

        super().__init__(
            model=model,
            n_models=n_models,
            subspace_size=subspace_size,
            training_method=training_method,
            lam=lam,
            drift_detector=drift_detector,
            warning_detector=warning_detector,
            disable_detector=disable_detector,
            disable_weighted_vote=disable_weighted_vote,
            seed=seed,
            metric=metric,
        )

        if aggregation_method not in {self._MEAN, self._MEDIAN}:
            raise ValueError(
                f"Invalid aggregation_method: {aggregation_method}.\n"
                f"Valid options are: {[self._MEAN, self._MEDIAN]}"
            )
        self.aggregation_method = aggregation_method

        if drift_detection_criteria not in {self._ERROR, self._PREDICTION}:
            raise ValueError(
                f"Invalid drift_detection_criteria: {drift_detection_criteria}.\n"
                f"Valid options are: {[self._ERROR, self._PREDICTION]}"
            )
        self.drift_detection_criteria = drift_detection_criteria

        self._base_learner_class = BaseSRPRegressor  # type: ignore

    def predict_one(self, x, **kwargs):
        y_pred = np.zeros(self.n_models)
        weights = np.ones(self.n_models)

        for i, model in enumerate(self.models):
            y_pred[i] = model.predict_one(x, **kwargs)
            if not self.disable_weighted_vote:
                metric_value = model.metric.get()
                weights[i] = metric_value if metric_value >= 0 else 0.0

        if self.aggregation_method == self._MEAN:
            if not self.disable_weighted_vote:
                if not self.metric.bigger_is_better:
                    # Invert weights so smaller values have larger influence
                    weights = -(weights - max(weights))
                if sum(weights) == 0:
                    # Average is undefined
                    return 0.0
            return np.average(y_pred, weights=weights)
        else:  # self.aggregation_method == self._MEDIAN:
            return np.median(y_pred)


class BaseSRPRegressor(BaseSRPEstimator):
    """Class representing the base learner of SRPClassifier."""

    def __init__(
        self,
        idx_original: int,
        model: base.Regressor,
        metric: RegressionMetric,
        created_on: int,
        drift_detector: base.DriftDetector,
        warning_detector: base.DriftDetector,
        is_background_learner,
        rng: random.Random,
        features=None,
        drift_detection_criteria: str | None = None,
    ):
        super().__init__(
            idx_original=idx_original,
            model=model,
            metric=metric,
            created_on=created_on,
            drift_detector=drift_detector,
            warning_detector=warning_detector,
            is_background_learner=is_background_learner,
            rng=rng,
            features=features,
        )
        self.drift_detection_criteria = drift_detection_criteria

    def learn_one(
        self,
        x: dict,
        y: base.typing.RegTarget,
        *,
        sample_weight: int,
        n_samples_seen: int,
        **kwargs,
    ):
        all_features = list(x.keys())
        if self.features is not None:
            # Select the subset of features to use
            x_subset = {k: x[k] for k in self.features if k in x}
        else:
            # Use all features
            x_subset = x

        # TODO Find a way to verify if the model natively supports sample_weight
        for _ in range(int(sample_weight)):
            self.model.learn_one(x=x_subset, y=y, **kwargs)

        # Drift detection input
        y_pred = self.model.predict_one(x_subset)
        if self.drift_detection_criteria == "error":
            # Track absolute error
            drift_detector_input = abs(y_pred - y)
        else:  # self.drift_detection_criteria == "prediction"
            # Track predicted target values
            drift_detector_input = y_pred

        if self._background_learner:
            # Train the background learner
            # Note: Pass the original instance x so features are correctly
            # selected based on the corresponding subspace
            self._background_learner.learn_one(
                x=x, y=y, sample_weight=sample_weight, n_samples_seen=n_samples_seen  # type: ignore
            )

        if not self.disable_drift_detector and not self.is_background_learner:
            # Check for warnings only if the background learner is active
            if not self.disable_background_learner:
                # Update the warning detection method
                self.warning_detector.update(drift_detector_input)
                # Check if there was a change
                if self.warning_detector.drift_detected:
                    self.n_warnings_detected += 1
                    self._trigger_warning(all_features=all_features, n_samples_seen=n_samples_seen)

            # ===== Drift detection =====
            # Update the drift detection method
            self.drift_detector.update(drift_detector_input)
            # Check if the was a change
            if self.drift_detector.drift_detected:
                self.n_drifts_detected += 1
                # There was a change, reset the model
                self.reset(all_features=all_features, n_samples_seen=n_samples_seen)

    def predict_one(self, x, **kwargs):
        # Select the features to use
        x_subset = {k: x[k] for k in self.features if k in x} if self.features is not None else x

        return self.model.predict_one(x_subset, **kwargs)
