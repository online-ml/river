import collections
import copy
import itertools
import math
import typing

import numpy as np

from river import base
from river.drift import ADWIN
from river.metrics import Accuracy
from river.metrics.base import MultiClassMetric
from river.tree import HoeffdingTreeClassifier


class SRPClassifier(base.WrapperMixin, base.EnsembleMixin, base.Classifier):
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
    nominal_attributes
        List of Nominal attributes. If empty, then assumes that all
        attributes are numerical. Note: Only applies if the base model
        allows to define the nominal attributes.
    seed
        Random number generator seed for reproducibility.
    metric
        Metric to track members performance within the ensemble.

    Examples
    --------
    >>> from river import synth
    >>> from river import ensemble
    >>> from river import tree
    >>> from river import evaluate
    >>> from river import metrics

    >>> dataset = synth.ConceptDriftStream(seed=42, position=500,
    ...                                    width=50).take(1000)
    >>> base_model = tree.HoeffdingTreeClassifier(
    ...     grace_period=50, split_confidence=0.01,
    ...     nominal_attributes=['age', 'car', 'zipcode']
    ... )
    >>> model = ensemble.SRPClassifier(
    ...     model=base_model, n_models=3, seed=42,
    ... )
    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)  # doctest: +SKIP
    Accuracy: 70.97%

    References
    ----------
    [^1]: Heitor Murilo Gomes, Jesse Read, Albert Bifet.
          Streaming Random Patches for Evolving Data Stream Classification.
          IEEE International Conference on Data Mining (ICDM), 2019.

    """

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
        model: base.Classifier = None,
        n_models: int = 100,
        subspace_size: typing.Union[int, float, str] = 0.6,
        training_method: str = "patches",
        lam: float = 6.0,
        drift_detector: base.DriftDetector = None,
        warning_detector: base.DriftDetector = None,
        disable_detector: str = "off",
        disable_weighted_vote: bool = False,
        nominal_attributes=None,
        seed=None,
        metric: MultiClassMetric = None,
    ):

        if model is None:
            model = HoeffdingTreeClassifier(grace_period=50, split_confidence=0.01)

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
        self.disable_detector = disable_detector

        if metric is None:
            metric = Accuracy()

        super().__init__([None])  # List of models is properly initialized later
        self.models = []
        self.model = model  # Not restricted to a specific base estimator.
        self.n_models = n_models
        self.subspace_size = subspace_size
        self.training_method = training_method
        self.lam = lam
        self.drift_detector = drift_detector
        self.warning_detector = warning_detector
        self.disable_weighted_vote = disable_weighted_vote
        self.metric = metric
        self.nominal_attributes = nominal_attributes if nominal_attributes else []
        self.seed = seed
        self._rng = np.random.default_rng(self.seed)

        self._n_samples_seen = 0
        self._subspaces = None

        self._base_learner_class = StreamingRandomPatchesBaseLearner

    @property
    def _wrapped_model(self):
        return self.model

    @classmethod
    def _unit_test_params(cls):
        return {"n_models": 3}

    def _unit_test_skips(self):
        return {"check_shuffle_features_no_impact"}

    def learn_one(self, x: dict, y: base.typing.ClfTarget, **kwargs):
        self._n_samples_seen += 1

        if not self.models:
            self._init_ensemble(list(x.keys()))

        for model in self.models:
            # Get prediction for instance
            y_pred = model.predict_proba_one(x)
            if y_pred:
                y_pred = max(y_pred, key=y_pred.get)
            else:
                y_pred = None  # Model is empty

            # Update performance evaluator
            model.metric.update(y_true=y, y_pred=y_pred)

            # Train using random subspaces without resampling,
            # i.e. all instances are used for training.
            if self.training_method == self._TRAIN_RANDOM_SUBSPACES:
                model.learn_one(
                    X=x,
                    y=y,
                    sample_weight=1.0,
                    n_samples_seen=self._n_samples_seen,
                    rng=self._rng,
                )
            # Train using random patches or resampling,
            # thus we simulate online bagging with Poisson(lambda=...)
            else:
                model.learn_one(
                    x=x,
                    y=y,
                    sample_weight=self._rng.poisson(lam=self.lam),
                    n_samples_seen=self._n_samples_seen,
                    rng=self._rng,
                )

        return self

    def predict_proba_one(self, x):

        y_pred = collections.Counter()

        if not self.models:
            self._init_ensemble(features=list(x.keys()))
            return y_pred

        for model in self.models:
            y_proba_temp = model.predict_proba_one(x)
            metric_value = model.metric.get()
            if not self.disable_weighted_vote and metric_value > 0.0:
                y_proba_temp = {
                    k: val * metric_value for k, val in y_proba_temp.items()
                }
            y_pred.update(y_proba_temp)

        total = sum(y_pred.values())
        if total > 0:
            return {label: proba / total for label, proba in y_pred.items()}
        return y_pred

    def _generate_subspaces(self, features: list):
        n_features = len(features)

        self._subspaces = [None] * n_features

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

            # Generate subspaces. The subspaces is a 2D array of shape
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

        subspace_indexes = np.arange(
            self.n_models
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
            self.models.append(
                self._base_learner_class(
                    idx_original=i,
                    model=self.model,
                    metric=self.metric,
                    created_on=self._n_samples_seen,
                    drift_detector=self.drift_detector,
                    warning_detector=self.warning_detector,
                    is_background_learner=False,
                    rng=self._rng,
                    features=subspace,
                    nominal_attributes=self.nominal_attributes,
                )
            )

    def reset(self):
        self.models = []
        self._n_samples_seen = 0
        self._rng = np.random.default_rng(self.seed)


class StreamingRandomPatchesBaseLearner:
    """
    Class representing the base learner of StreamingRandomPatchesClassifier.
    """

    def __init__(
        self,
        idx_original: int,
        model: base.Classifier,
        metric: MultiClassMetric,
        created_on: int,
        drift_detector: base.DriftDetector,
        warning_detector: base.DriftDetector,
        is_background_learner,
        rng: np.random.Generator,
        features=None,
        nominal_attributes=None,
    ):
        self.idx_original = idx_original
        self.created_on = created_on
        self.model = model.clone()
        self.metric = copy.deepcopy(metric)
        # Make sure that the metric is not initialized, e.g. when creating background learners.
        self.metric.cm.reset()

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

        # Background learner
        self._background_learner = (
            None
        )  # type: typing.Optional[StreamingRandomPatchesBaseLearner]
        self._background_learner_class = StreamingRandomPatchesBaseLearner

        # Nominal attributes
        self.nominal_attributes = nominal_attributes
        self._set_nominal_attributes = self._can_set_nominal_attributes()

        # Random number generator (initialized)
        self.rng = rng

    def learn_one(
        self,
        x: dict,
        y: base.typing.ClfTarget,
        *,
        sample_weight: int,
        n_samples_seen: int,
        rng: np.random.Generator,
    ):
        all_features = [feature for feature in x.keys()]
        if self.features is not None:
            # Select the subset of features to use
            x_subset = {k: x[k] for k in self.features}
            if self._set_nominal_attributes and hasattr(
                self.model, "nominal_attributes"
            ):
                self.model.nominal_attributes = list(
                    set(self.features).intersection(set(self.nominal_attributes))
                )
                self._set_nominal_attributes = False
        else:
            # Use all features
            x_subset = x

        # TODO Find a way to verify if the model natively supports sample_weight
        for _ in range(sample_weight):
            self.model.learn_one(x=x_subset, y=y)

        if self._background_learner:
            # Train the background learner
            # Note: Pass the original instance x so features are correctly
            # selected based on the corresponding subspace
            self._background_learner.learn_one(
                x=x,
                y=y,
                sample_weight=sample_weight,
                n_samples_seen=n_samples_seen,
                rng=rng,
            )

        if not self.disable_drift_detector and not self.is_background_learner:
            correctly_classifies = self.model.predict_one(x_subset) == y
            # Check for warnings only if the background learner is active
            if not self.disable_background_learner:
                # Update the warning detection method
                self.warning_detector.update(int(not correctly_classifies))
                # Check if there was a change
                if self.warning_detector.change_detected:
                    self.n_warnings_detected += 1
                    self._trigger_warning(
                        all_features=all_features,
                        n_samples_seen=n_samples_seen,
                        rng=rng,
                    )

            # ===== Drift detection =====
            # Update the drift detection method
            self.drift_detector.update(int(not correctly_classifies))
            # Check if the was a change
            if self.drift_detector.change_detected:
                self.n_drifts_detected += 1
                # There was a change, reset the model
                self.reset(
                    all_features=all_features, n_samples_seen=n_samples_seen, rng=rng
                )

    def predict_proba_one(self, x):
        # Select the features to use
        x_subset = {k: x[k] for k in self.features} if self.features is not None else x

        return self.model.predict_proba_one(x_subset)

    def _trigger_warning(
        self, all_features, n_samples_seen: int, rng: np.random.Generator
    ):
        # Randomly generate a new subspace from all the original features
        subspace = (
            None
            if self.features is None
            else random_subspace(
                all_features=all_features, k=len(self.features), rng=rng
            )
        )

        # Initialize the background learner
        self._background_learner = self._background_learner_class(
            idx_original=self.idx_original,
            model=self.model,
            metric=self.metric,
            created_on=n_samples_seen,
            drift_detector=self.drift_detector,
            warning_detector=self.warning_detector,
            is_background_learner=True,
            rng=self.rng,
            features=subspace,
            nominal_attributes=self.nominal_attributes,
        )
        # Hard-reset the warning method
        self.warning_detector = self.warning_detector.clone()

    def reset(self, all_features: list, n_samples_seen: int, rng: np.random.Generator):
        # Randomly generate a new subspace from all the original features
        subspace = (
            None
            if self.features is None
            else random_subspace(
                all_features=all_features, k=len(self.features), rng=rng
            )
        )

        if not self.disable_background_learner and self._background_learner is not None:
            # Replace model with the corresponding background model
            self.model = self._background_learner.model
            self.drift_detector = self._background_learner.drift_detector
            self.warning_detector = self._background_learner.warning_detector
            self.metric = self._background_learner.metric
            self.metric.cm.reset()
            self.created_on = self._background_learner.created_on
            self.features = self._background_learner.features
            self._background_learner = None
        else:
            # Randomly generate a new subspace from all the original features
            subspace = (
                None
                if self.features is None
                else random_subspace(
                    all_features=all_features, k=len(self.features), rng=rng
                )
            )
            # Reset model
            self.model = self.model.clone()
            self.metric.cm.reset()
            self.created_on = n_samples_seen
            self.drift_detector = self.drift_detector.clone()
            self.features = subspace
            self._set_nominal_attributes = self._can_set_nominal_attributes()

    def _can_set_nominal_attributes(self):
        return self.nominal_attributes is not None and len(self.nominal_attributes) > 0


def random_subspace(all_features: list, k: int, rng: np.random.Generator):
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
    return rng.choice(all_features, k, replace=False)
