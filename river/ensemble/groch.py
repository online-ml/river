from __future__ import annotations

import collections
import copy
import typing

import numpy as np

from river import base, cluster, drift, tree, utils
from river.utils.math import mahalanobis_distance

__all__ = ["GroCHClassifier", "LocalLearner", "TopologyDistillator"]


class TopologyDistillator(base.Base):
    """Encapsulates a clustering/topology learner (e.g., `cluster.GNG`) that summarizes
    data distributions into prototype vectors.

    Parameters
    ----------
    clusterer
        Incremental clusterer used to learn prototypes. Defaults to
        `cluster.GNG(lambda_=5, max_age=200)`.
    feature_subset
        Optional list of feature names used for topology learning.
    include_target
        If `True`, append the numeric target `y` as the final dimension of each prototype vector.
    train_online
        If `True`, update `clusterer` immediately on each `learn_one` call. If `False`,
        buffer incoming instances and train lazily via `train_from_buffer()`.

    """

    def __init__(
        self,
        clusterer: base.Clusterer | None = None,
        feature_subset: list[str | int] | None = None,
        include_target: bool = False,
        train_online: bool = True,
    ):
        self.clusterer = clusterer if clusterer is not None else cluster.GNG(lambda_=5, max_age=200)
        self.feature_subset = feature_subset
        self.include_target = include_target
        self.train_online = train_online
        self._buffer: list[tuple[dict, typing.Any]] = []

    def _select_features(self, x: dict) -> dict:
        if self.feature_subset:
            return {k: x[k] for k in self.feature_subset if k in x}
        return x

    def _learn_single(self, x: dict, y=None) -> None:
        if self.feature_subset is not None and hasattr(self.clusterer, "learn_array"):
            pattern = [float(x.get(k, 0.0)) for k in self.feature_subset]
            if self.include_target:
                pattern.append(float(y) if y is not None else 0.0)
            self.clusterer.learn_array(pattern)
        else:
            x_sub = self._select_features(x)
            if self.include_target and y is not None:
                x_sub = dict(x_sub)
                x_sub["__target__"] = float(y)
            self.clusterer.learn_one(x_sub)

    def learn_one(self, x: dict, y=None) -> TopologyDistillator:
        if self.train_online:
            self._learn_single(x, y)
        else:
            self._buffer.append((dict(x), y))
        return self

    def train_from_buffer(self) -> TopologyDistillator:
        if not self._buffer:
            return self
        for x, y in self._buffer:
            self._learn_single(x, y)
        self._buffer.clear()
        return self

    @property
    def n_prototypes_created(self) -> int:
        self.train_from_buffer()
        if hasattr(self.clusterer, "n_prototypes_created"):
            return int(self.clusterer.n_prototypes_created)
        return len(self.prototypes)

    @property
    def prototypes_array(self) -> np.ndarray:
        self.train_from_buffer()
        if hasattr(self.clusterer, "prototypes_array"):
            arr = self.clusterer.prototypes_array
            if len(arr) > 0:
                return arr
            d = (len(self.feature_subset) if self.feature_subset else 0) + (
                1 if self.include_target else 0
            )
            return np.zeros((0, d), dtype=np.float64)
        protos = self.prototypes
        if not protos:
            return np.zeros((0, 0), dtype=np.float64)
        keys = self.feature_subset if self.feature_subset else sorted(protos[0].keys())
        return np.array([[float(p.get(k, 0.0)) for k in keys] for p in protos], dtype=np.float64)

    @property
    def prototypes(self) -> list[dict]:
        self.train_from_buffer()
        if hasattr(self.clusterer, "centers"):
            centers = self.clusterer.centers
            return list(centers.values()) if isinstance(centers, dict) else list(centers)
        return []

    @prototypes.setter
    def prototypes(self, values: list[dict]) -> None:
        if hasattr(self.clusterer, "centers"):
            try:
                self.clusterer.centers = values
            except AttributeError:
                pass

    def clone(
        self, new_params: dict | None = None, include_attributes: bool = False
    ) -> TopologyDistillator:
        """Return a deep copy preserving the trained clusterer topology."""
        if new_params is not None:
            return super().clone(new_params=new_params, include_attributes=include_attributes)
        if not self.train_online:
            self.train_from_buffer()
        cloned = TopologyDistillator(
            clusterer=copy.deepcopy(self.clusterer),
            feature_subset=self.feature_subset,
            include_target=self.include_target,
            train_online=self.train_online,
        )
        cloned._buffer = list(self._buffer)
        return cloned

    def __deepcopy__(self, memo) -> TopologyDistillator:
        cloned = self.clone()
        memo[id(self)] = cloned
        return cloned


class LocalLearner(base.Base):
    """Wraps a River classifier with Concept History lifecycle metadata.

    Parameters
    ----------
    model
        Base River classifier instance wrapped by this learner.
    created_at
        Stream timestamp (sample index) at which the learner was created.

    """

    def __init__(self, model: base.Classifier, created_at: int = 0):
        self.model = model
        self.usage: int = 0
        self.created_at: int = created_at
        self.last_time_used: float = float(created_at)
        self.instances_trained: int = 0
        self.cum_evaluated: int = 0
        self.cum_correct: int = 0
        self._w_results: collections.deque | None = None

    def predict_one(self, x: dict):
        return self.model.predict_one(x)

    def predict_proba_one(self, x: dict):
        return self.model.predict_proba_one(x)

    def learn_one(self, x: dict, y: base.typing.ClfTarget) -> LocalLearner:
        self.instances_trained += 1
        self.model.learn_one(x, y)
        return self

    def clone(
        self, new_params: dict | None = None, include_attributes: bool = False
    ) -> LocalLearner:
        """Return a deep copy preserving the trained model state and metadata."""
        if new_params is not None:
            return super().clone(new_params=new_params, include_attributes=include_attributes)
        cloned = LocalLearner(copy.deepcopy(self.model), self.created_at)
        cloned.usage = self.usage
        cloned.last_time_used = self.last_time_used
        cloned.instances_trained = self.instances_trained
        cloned.cum_evaluated = self.cum_evaluated
        cloned.cum_correct = self.cum_correct
        return cloned

    def __deepcopy__(self, memo) -> LocalLearner:
        cloned = self.clone()
        memo[id(self)] = cloned
        return cloned


class GroCHClassifier(base.Classifier):
    """Growing Concept History (GroCH) classifier for recurring concept drifts.

    GroCH [^1] maintains an active foreground classifier, a background classifier
    trained during drift warnings, and a Growing Concept History (`CH`) of groups
    indexed by topological prototypes (e.g., Growing Neural Gas) to detect and
    reuse recurring concepts.

    Parameters
    ----------
    model
        Base classifier template. Defaults to `tree.HoeffdingTreeClassifier(leaf_prediction="nba")`.
    sigma
        Maximum prototype-to-group distance threshold for matching a concept group.
    omega
        Maximum number of historical classifiers stored per concept group.
    mu
        Maximum warning window length before forcing a warning window reset.
    beta_stability
        Minimum number of instances the active classifier must be trained on before
        it is eligible for insertion into `CH` upon a normal drift.
    theta
        Minimum warning window length required for a normal drift (below `theta` is an early drift).
    beta_penalty
        Multiplicative weight penalty factor used when `weighted_tests_on_warn=True`.
    drift_detector
        Concept drift detector. Defaults to `drift.binary.RDDM()`.
    warning_detector
        Optional warning detector. If `None`, shares `drift_detector` when it exposes
        `warning_detected`, otherwise defaults to `drift.ADWIN(delta=0.01)`.
    distillator
        Optional pre-configured `TopologyDistillator` template.
    clusterer
        Optional `base.Clusterer` used to build default distillators.
    feature_subset
        Optional list of feature names used for topology distillation and similarity.
    pre_training_data
        Optional list of `(dataset, base_model)` tuples used to pre-populate `CH`.
    dist_metric
        Concept similarity metric: `"mahalanobis"` or `"euclidean"`.
    inv_cov
        Optional pre-computed precision matrix (inverse covariance) as a 2D `numpy.ndarray`
        when `dist_metric="mahalanobis"`.
    gng_lambda
        GNG insertion interval when constructing the default `cluster.GNG`.
    gng_max_age
        GNG maximum edge age when constructing the default `cluster.GNG`.
    gng_max_nodes
        Optional maximum number of GNG nodes.
    evaluator_width
        Sliding evaluation window size during warnings. Defaults to `mu`.
    always_insert
        If `True`, insert the active classifier into `GC` even when `GC == GH` and
        error does not improve upon existing models in `GC`.
    force_early_drifts
        If `True`, evaluate active vs background transitions even when `len(W) < theta`.
    insert_on_early_drift
        If `True`, allow inserting the active classifier into `CH` on early drifts.
    always_reset_warning_at_early_drift
        If `True`, reset the warning window before initializing background training on early drifts.
    save_classifiers_on_false_alarm
        If `True`, save the warning-onset snapshot `_cTemp` into `CH` on false alarms.
    train_on_warn
        If `True`, train the background classifier `b` during warnings.
    train_active_on_warn
        If `True`, keep training the active classifier `c` during warnings. Defaults to `train_on_warn`.
    train_after_drift_evaluation
        If `True`, update `c`, `b`, `W`, and topology after drift detection rather than before.
    train_recurring_on_w
        If `True`, replay warning window `W` on a retrieved recurring classifier upon transition.
    weighted_tests_on_warn
        If `True`, switch predictions during warnings to `b` whenever `b_weight > c_weight`.
    test_with_bkg_on_warn
        If `True`, predict with `b` during warnings once `len(W) >= theta`.
    end_of_warning_after_warning_zone
        If `True`, reset the warning window as soon as the detector exits its warning zone.
    priority_of_recurring
        Tie-breaking policy among equal-error recurring candidates: `"new"` or `"old"`.
    group_replacement_policy
        Eviction policy when a group reaches `omega` classifiers: `"fifo"` or `"least used model"`.
    min_topology_size_for_drift
        Minimum number of created prototypes required before inserting into `CH`.
    active_better_when_bkg_nan
        If `True`, prefer the active classifier (using cumulative accuracy fallback) when
        the background classifier has no window evaluations (`NaN`).
    update_group_topologies
        If `True`, replace the matched group's stored topology with the active topology on insertion.
    multi_cluster
        If `True`, evaluate classifiers across all groups within distance `< sigma`.
    custom_distance_func
        Optional custom pairwise distance callable `(p1, p2) -> float`.
    custom_mean_nn_dist_func
        Optional custom set-to-set mean nearest-neighbor distance callable `(P, PG) -> float`.
    log_level
        Console observability level: `0` (silent), `1` (lifecycle & drifts),
        `2` (CH candidates), `3` (model details).

    Examples
    --------
    >>> from river import datasets, ensemble, metrics
    >>> model = ensemble.GroCHClassifier(log_level=0)
    >>> metric = metrics.Accuracy()
    >>> for x, y in datasets.Phishing().take(100):
    ...     y_pred = model.predict_one(x)
    ...     if y_pred is not None:
    ...         metric.update(y, y_pred)
    ...     model.learn_one(x, y)
    >>> metric.get() > 0.5
    True

    References
    ----------
    [^1]: Suárez-Cetrulo, A. L., Cervantes, A., & Quintana, D. (2026).
          A growing concept history of recurring classifiers for high-frequency data streams.
          Neurocomputing, 684, 133506. https://doi.org/10.1016/j.neucom.2026.133506

    """

    def __init__(
        self,
        model: base.Classifier | None = None,
        sigma: float = 1.5,
        omega: int = 10,
        mu: int = 500,
        beta_stability: int = 200,
        theta: int = 15,
        beta_penalty: float = 0.5,
        drift_detector: base.DriftDetector | None = None,
        warning_detector: base.DriftDetector | None = None,
        distillator: TopologyDistillator | None = None,
        clusterer: base.Clusterer | None = None,
        feature_subset: list[str | int] | None = None,
        pre_training_data: list | None = None,
        dist_metric: str = "mahalanobis",
        inv_cov: np.ndarray | None = None,
        gng_lambda: int = 5,
        gng_max_age: int = 200,
        gng_max_nodes: int | None = None,
        evaluator_width: int | None = None,
        always_insert: bool = False,
        force_early_drifts: bool = False,
        insert_on_early_drift: bool = False,
        always_reset_warning_at_early_drift: bool = False,
        save_classifiers_on_false_alarm: bool = False,
        train_on_warn: bool = True,
        train_active_on_warn: bool | None = None,
        train_after_drift_evaluation: bool = False,
        train_recurring_on_w: bool = True,
        weighted_tests_on_warn: bool = False,
        test_with_bkg_on_warn: bool = False,
        end_of_warning_after_warning_zone: bool = False,
        priority_of_recurring: str = "new",
        group_replacement_policy: str = "fifo",
        min_topology_size_for_drift: int = 1,
        active_better_when_bkg_nan: bool = False,
        update_group_topologies: bool = False,
        multi_cluster: bool = False,
        custom_distance_func: typing.Callable | None = None,
        custom_mean_nn_dist_func: typing.Callable | None = None,
        log_level: int = 0,
    ):
        self.model = (
            model if model is not None else tree.HoeffdingTreeClassifier(leaf_prediction="nba")
        )
        self.sigma = sigma
        self.omega = omega
        self.mu = mu
        self.beta_stability = beta_stability
        self.theta = theta
        self.beta_penalty = beta_penalty
        self.drift_detector = drift_detector if drift_detector is not None else drift.binary.RDDM()
        self.warning_detector = warning_detector
        self.distillator = distillator
        self.clusterer = clusterer
        self.feature_subset = feature_subset
        self.pre_training_data = pre_training_data
        self.dist_metric = dist_metric
        self.inv_cov = inv_cov
        self.gng_lambda = gng_lambda
        self.gng_max_age = gng_max_age
        self.gng_max_nodes = gng_max_nodes
        self.evaluator_width = evaluator_width
        self.always_insert = always_insert
        self.force_early_drifts = force_early_drifts
        self.insert_on_early_drift = insert_on_early_drift
        self.always_reset_warning_at_early_drift = always_reset_warning_at_early_drift
        self.save_classifiers_on_false_alarm = save_classifiers_on_false_alarm
        self.train_on_warn = train_on_warn
        self.train_active_on_warn = train_active_on_warn
        self.train_after_drift_evaluation = train_after_drift_evaluation
        self.train_recurring_on_w = train_recurring_on_w
        self.weighted_tests_on_warn = weighted_tests_on_warn
        self.test_with_bkg_on_warn = test_with_bkg_on_warn
        self.end_of_warning_after_warning_zone = end_of_warning_after_warning_zone
        self.priority_of_recurring = priority_of_recurring
        self.group_replacement_policy = group_replacement_policy
        self.min_topology_size_for_drift = min_topology_size_for_drift
        self.active_better_when_bkg_nan = active_better_when_bkg_nan
        self.update_group_topologies = update_group_topologies
        self.multi_cluster = multi_cluster
        self.custom_distance_func = custom_distance_func
        self.custom_mean_nn_dist_func = custom_mean_nn_dist_func
        self.log_level = log_level

        # Resolved internal settings (keep __init__ params untouched for idempotent clone())
        self._evaluator_width: int = evaluator_width if evaluator_width is not None else mu
        self._train_active_on_warn: bool = (
            train_active_on_warn if train_active_on_warn is not None else train_on_warn
        )

        # Runtime detectors
        self._drift_detector = self.drift_detector.clone()
        if warning_detector is None:
            if hasattr(self._drift_detector, "warning_detected"):
                self._warning_detector = self._drift_detector
            else:
                self._warning_detector = drift.ADWIN(delta=0.01)
        elif warning_detector is drift_detector:
            self._warning_detector = self._drift_detector
        else:
            self._warning_detector = warning_detector.clone()

        # Mahalanobis / normalization state
        self._inv_cov: np.ndarray | None = inv_cov
        self._mu_vec: np.ndarray | None = None
        self._feature_keys: list | None = None
        self._det_root: float = 1.0
        self._min_dict: dict | None = None
        self._max_dict: dict | None = None

        # Active topology & learners
        self._distillator_template = distillator.clone() if distillator is not None else None
        self._active_distillator = (
            distillator if distillator is not None else self._create_distillator()
        )
        self.c: LocalLearner = LocalLearner(self.model.clone())
        self.b: LocalLearner | None = LocalLearner(self.model.clone())
        self.CH: list[dict] = []
        self.GH: dict | None = None
        self.GC: dict | None = None

        # Warning & drift tracking state
        self.c_weight: float = 1.0
        self.b_weight: float = 1.0
        self.W: list[tuple[dict, typing.Any]] | None = None
        self.n_samples_seen: int = 0
        self.last_drift_at: int = -1
        self.last_warning_at: int = 0
        self.last_drift_was_false_alarm: bool = False
        self.last_drift_was_early: bool = False
        self.last_drift_was_historical: bool = False
        self.last_drift_was_inserted: bool = False
        self.last_drift_retrieved_group: int | None = None
        self._prev_warning_zone: bool = False
        self._c_temp: LocalLearner | None = None

        if pre_training_data:
            self.pre_populate(pre_training_data)

    @property
    def _multiclass(self) -> bool:
        return self.model._multiclass

    @classmethod
    def _unit_test_params(cls):
        yield {"log_level": 0}

    def _unit_test_skips(self) -> set[str]:
        # Default HoeffdingTreeClassifier(leaf_prediction="nba") can underflow
        # Gaussian NB on 19-feature ImageSegments when feature order changes.
        return {"check_shuffle_features_no_impact"}

    def _log(self, msg: str, level: int = 1) -> None:
        if self.log_level >= level:
            print(f"[GroCH t={self.n_samples_seen}] {msg}", flush=True)

    @staticmethod
    def _describe_model(learner: LocalLearner | base.Classifier) -> str:
        m = learner.model if isinstance(learner, LocalLearner) else learner
        if hasattr(m, "n_nodes"):
            return f"{m.__class__.__name__} (nodes={m.n_nodes})"
        return str(m)

    @staticmethod
    def _window_error(learner: LocalLearner | None, fallback_cum: bool = False) -> float:
        if learner is None:
            return float("nan")
        res = learner._w_results
        if not res:
            if fallback_cum and learner.cum_evaluated > 0:
                return 1.0 - (learner.cum_correct / learner.cum_evaluated)
            return float("nan")
        return 1.0 - (sum(res) / len(res))

    def _normalize_x(self, x: dict) -> dict:
        if self._min_dict is None or self._max_dict is None:
            return x
        x_norm = {}
        for k, v in x.items():
            if k in self._min_dict:
                min_v = self._min_dict[k]
                max_v = self._max_dict[k]
                x_norm[k] = (v - min_v) / (max_v - min_v) if max_v > min_v else 0.0
            else:
                x_norm[k] = v
        return x_norm

    def _compute_normalization_params(self, datasets: list) -> None:
        if self.feature_subset:
            self._feature_keys = sorted(self.feature_subset)
        elif not self._feature_keys:
            if not datasets:
                self._feature_keys = []
                return
            first_x = datasets[0][0][0][0]
            self._feature_keys = sorted(first_x.keys())

        n_attributes = len(self._feature_keys)
        self._min_dict = {}
        self._max_dict = {}
        for k in self._feature_keys:
            vals = [x.get(k, 0.0) for dataset, _ in datasets for x, _ in dataset]
            self._min_dict[k] = min(vals) if vals else 0.0
            self._max_dict[k] = max(vals) if vals else 1.0

        if self.dist_metric.lower() == "mahalanobis":
            all_x = [
                [self._normalize_x(x).get(k, 0.0) for k in self._feature_keys]
                for dataset, _ in datasets
                for x, _ in dataset
            ]
            if not all_x:
                return
            data = np.array(all_x, dtype=np.float64)
            cov = np.cov(data, rowvar=False, ddof=1)
            self._inv_cov = np.linalg.pinv(cov)
            self._mu_vec = np.mean(data, axis=0)
            det = abs(np.linalg.det(cov))
            self._det_root = float(np.power(det, 1.0 / n_attributes)) if n_attributes > 0 else 1.0
            self._log(f"Mahalanobis Params: data_shape={data.shape}, det_root={self._det_root:.4f}")
        else:
            self._log("Euclidean/Minkowski Params initialized.")

    def _create_distillator(self) -> TopologyDistillator:
        if self._distillator_template is not None:
            return self._distillator_template.clone()
        if self.clusterer is not None:
            c = self.clusterer.clone()
            if isinstance(c, cluster.GNG) and self._inv_cov is not None:
                c.dist_metric = self.dist_metric
                c.inv_cov = self._inv_cov
            return TopologyDistillator(clusterer=c, feature_subset=self.feature_subset)

        gng = cluster.GNG(
            lambda_=self.gng_lambda,
            max_age=self.gng_max_age,
            max_nodes=self.gng_max_nodes,
            dist_metric=self.dist_metric,
            inv_cov=self._inv_cov,
        )
        return TopologyDistillator(clusterer=gng, feature_subset=self.feature_subset)

    def pre_populate(self, pre_training_data: list) -> None:
        """Pre-populate the Concept History with initial `(dataset, base_model)` concepts."""
        self._log(f"Pre-populating CH with {len(pre_training_data)} concepts")
        self._compute_normalization_params(pre_training_data)

        for g_id, (dataset, base_model) in enumerate(pre_training_data):
            dist = self._create_distillator()
            learner = LocalLearner(base_model.clone())
            n_trained = 0
            for x, y in dataset:
                dist.learn_one(self._normalize_x(x))
                learner.learn_one(x, str(y))
                n_trained += 1
            self.CH.append(
                {
                    "prototypes": dist.prototypes,
                    "classifiers": [learner],
                    "distillator": copy.deepcopy(dist),
                    "group_id": g_id,
                }
            )
            self._log(f"Pre-populated model with {n_trained} samples")
        self._log(f"CH size after pre-population: {len(self.CH)}")

    # ------------------------------------------------------------------
    # Warning window & drift handling helpers
    # ------------------------------------------------------------------

    def _clear_window_results(self) -> None:
        if self.c._w_results is not None:
            self.c._w_results.clear()
        if self.b is not None and self.b._w_results is not None:
            self.b._w_results.clear()
        for group in self.CH:
            for learner in group["classifiers"]:
                if learner._w_results is not None:
                    learner._w_results.clear()

    def _reset_warning_window(self, starting_ww: bool = True) -> None:
        if starting_ww:
            self.b = None
            self._c_temp = None
            self._clear_window_results()
            if self.W is not None:
                for wx, wy in self.W:
                    if not self._train_active_on_warn:
                        self.c.learn_one(wx, wy)
                    self._active_distillator.learn_one(self._normalize_x(wx), wy)

        self.W = None
        if getattr(self._warning_detector, "_is_addm", False) or (
            "ADDM" in self._warning_detector.__class__.__name__
        ):
            self._warning_detector = self._warning_detector.clone()
        self._prev_warning_zone = False

    def _start_warning_window(self) -> None:
        if self.W is not None:
            self._reset_warning_window(starting_ww=True)

        self.last_warning_at = self.n_samples_seen
        self._log("WARNING DETECTED")
        self.W = []
        self._c_temp = self.c.clone()
        self._c_temp.last_time_used = float(self.n_samples_seen)
        self._c_temp.usage = 0
        self.b = LocalLearner(self.model.clone(), self.n_samples_seen)
        self.c_weight = 1.0
        self.b_weight = 1.0

        self.c._w_results = collections.deque(maxlen=self._evaluator_width)
        self.b._w_results = collections.deque(maxlen=self._evaluator_width)
        for group in self.CH:
            for learner in group["classifiers"]:
                learner._w_results = collections.deque(maxlen=self._evaluator_width)

    def _evaluate_warning_window(self, x: dict, y: base.typing.Target) -> None:
        if self.W is None or self.b is None:
            return
        y_str = str(y)
        for learner in (self.c, self.b):
            if learner._w_results is None:
                learner._w_results = collections.deque(maxlen=self._evaluator_width)
            y_pred = learner.predict_one(x)
            learner._w_results.append(1 if (y_pred is not None and y_str == str(y_pred)) else 0)

        for group in self.CH:
            for learner in group["classifiers"]:
                if learner._w_results is None:
                    learner._w_results = collections.deque(maxlen=self._evaluator_width)
                y_pred = learner.predict_one(x)
                learner._w_results.append(1 if (y_pred is not None and y_str == str(y_pred)) else 0)

    def _update_weights(self, x: dict, y: base.typing.ClfTarget) -> None:
        y_str = str(y)
        for learner, weight_attr in ((self.c, "c_weight"), (self.b, "b_weight")):
            if learner is None:
                continue
            correct = str(learner.predict_one(x)) == y_str
            current = getattr(self, weight_attr)
            setattr(
                self,
                weight_attr,
                current / self.beta_penalty if correct else current * self.beta_penalty,
            )

    def _find_or_create_group(self, prototypes) -> dict:
        group = self._find_group(prototypes)
        if group is not None:
            self._log("Matched Existing Group in CH for GC")
            return group

        self._log("Creating NEW Group in CH")
        new_dist = (
            self._active_distillator.clone()
            if hasattr(self._active_distillator, "clone")
            else copy.deepcopy(self._active_distillator)
        )
        g_id = len(self.CH)
        if hasattr(prototypes, "distillator"):
            prototypes.distillator = new_dist
        if hasattr(prototypes, "group_id"):
            prototypes.group_id = g_id
        group = {
            "prototypes": prototypes,
            "classifiers": [],
            "distillator": new_dist,
            "group_id": g_id,
        }
        self.CH.append(group)
        return group

    def _insert_classifier_to_group(self, group: dict, source_learner: LocalLearner) -> None:
        if len(group["classifiers"]) >= self.omega:
            self._remove_classifier(group)
        g_id = group.get("group_id")
        self._log(
            f"INSERTING FOREGROUND MODEL into Group {g_id} (size after: {len(group['classifiers']) + 1})",
            level=1,
        )
        snapshot = source_learner.clone()
        snapshot.last_time_used = float(self.last_warning_at)
        snapshot.usage = 0
        snapshot._w_results = collections.deque(maxlen=self._evaluator_width)
        group["classifiers"].append(snapshot)
        if self.log_level >= 2:
            g_models = [
                f"#{idx}(c_at={m.created_at}, ltu={m.last_time_used}, usage={m.usage})"
                for idx, m in enumerate(group["classifiers"])
            ]
            self._log(f"  Group {g_id} classifiers: {', '.join(g_models)}", level=2)
        if self.log_level >= 3:
            self._log(f"  Inserted classifier structure: {self._describe_model(snapshot)}", level=3)

    def _insert_into_ch(self, check_error_improvement: bool = True) -> None:
        prototypes: list[dict] | np.ndarray
        if self.custom_mean_nn_dist_func is not None:
            prototypes = self._active_distillator.prototypes
            n_protos = getattr(self._active_distillator, "n_prototypes_created", len(prototypes))
        else:
            prototypes = self._active_distillator.prototypes_array
            n_protos = self._active_distillator.n_prototypes_created
        if n_protos < self.min_topology_size_for_drift:
            self._log("There weren't enough prototypes in the topology.", level=1)
            self.last_drift_was_inserted = False
            return

        self.GC = self._find_or_create_group(prototypes)

        should_add = True
        if check_error_improvement:
            err_c = self._window_error(self.c)
            valid_gc_errs = [
                e
                for e in (self._window_error(m) for m in self.GC["classifiers"])
                if not np.isnan(e)
            ]
            min_gc_err = min(valid_gc_errs) if valid_gc_errs else 1.0
            err_c_improves_gc = (not np.isnan(err_c)) and (err_c <= min_gc_err)
            should_add = self.always_insert or (self.GC is not self.GH) or err_c_improves_gc

        if should_add:
            source = self._c_temp if self._c_temp is not None else self.c
            self._insert_classifier_to_group(self.GC, source)
            self.last_drift_was_inserted = True
        else:
            self._log(
                "SKIPPING INSERTION: Error is not lower than best in identical group", level=1
            )
            self.last_drift_was_inserted = False

        if self.update_group_topologies and self.GC is not None:
            g_id = self.GC.get("group_id")
            self._log(f"Updating group {g_id} topology with active topology", level=1)
            new_top = (
                self._active_distillator.clone()
                if hasattr(self._active_distillator, "clone")
                else copy.deepcopy(self._active_distillator)
            )
            self.GC["distillator"] = new_top
            self.GC["prototypes"] = (
                new_top.prototypes
                if self.custom_mean_nn_dist_func is not None
                else new_top.prototypes_array
            )

    def _handle_drift(self, x: dict, y: base.typing.ClfTarget) -> None:
        self._log(f"DRIFT DETECTED. Current CH size: {len(self.CH)}")
        is_early_drift = self.W is None or len(self.W) < self.theta

        if not is_early_drift or self.force_early_drifts:
            if is_early_drift and (
                self.W is None or len(self.W) < 1 or self.always_reset_warning_at_early_drift
            ):
                self._start_warning_window()
                assert self.b is not None
                self.b.learn_one(x, y)
            elif self.b is None:
                self.b = LocalLearner(self.model.clone(), self.n_samples_seen)
                if self.train_on_warn:
                    self.b.learn_one(x, y)
                self.b._w_results = collections.deque(maxlen=self._evaluator_width)

            assert self.b is not None
            if is_early_drift:
                self.GH = None
                gh_models: list[LocalLearner] = []
                gh_prototypes: typing.Any = []
            else:
                w_norm = [(self._normalize_x(wx), wy) for wx, wy in self.W] if self.W else []
                p_to_match = (
                    self.W if (self.custom_mean_nn_dist_func is not None and self.W) else w_norm
                )
                if self.multi_cluster:
                    matched_groups = self._find_groups(p_to_match)
                    self.GH = (
                        self._rank_concept_history_groups(matched_groups)
                        if matched_groups
                        else None
                    )
                    if self.GH:
                        self._log(
                            f"Matched Historical Group for GH (MultiCluster): Group {self.GH.get('group_id')}"
                        )
                else:
                    self.GH = self._find_group(p_to_match)
                    if self.GH:
                        self._log(f"Matched Historical Group for GH (dist < {self.sigma})")
                gh_models = self.GH["classifiers"] if self.GH else []
                gh_prototypes = self.GH["prototypes"] if self.GH else []

            new_learner, td, was_false_alarm, best_gh_idx = self._select_drift(
                self.c, self.b, gh_models, gh_prototypes
            )
            self.last_drift_was_false_alarm = was_false_alarm
            self.last_drift_was_early = is_early_drift
            self.last_drift_was_historical = best_gh_idx is not None
            self.last_drift_retrieved_group = (
                self.GH.get("group_id", getattr(self.GH.get("prototypes"), "group_id", None))
                if (self.GH and best_gh_idx is not None)
                else None
            )

            # Capture retrieved group topology before _insert_into_ch potentially updates it
            new_distillator = None
            if not was_false_alarm and best_gh_idx is not None and self.GH:
                gh_dist = self.GH.get("distillator")
                if gh_dist is not None:
                    new_distillator = (
                        gh_dist.clone() if hasattr(gh_dist, "clone") else copy.deepcopy(gh_dist)
                    )

            if is_early_drift:
                if was_false_alarm:
                    self._log("EARLY_DRIFT_1 - FALSE ALARM (Active error <= BKG error)")
                else:
                    self._log(
                        "DRIFT TO BKG MODEL (Early drift confirmed, BKG error < Active error)"
                    )

            if was_false_alarm:
                should_insert = bool(
                    self.save_classifiers_on_false_alarm and self._c_temp is not None
                )
            elif is_early_drift:
                should_insert = bool(self.insert_on_early_drift)
            else:
                should_insert = bool(self.c.instances_trained > self.beta_stability)
            self.last_drift_was_inserted = should_insert

            if should_insert:
                self._insert_into_ch(check_error_improvement=True)

            if not was_false_alarm:
                if best_gh_idx is not None and self.GH:
                    if self.train_recurring_on_w and self.W is not None:
                        for wx, wy in self.W:
                            new_learner.learn_one(wx, wy)
                    self.GH["classifiers"][best_gh_idx].usage += 1

                self.c = new_learner
                self.c.instances_trained = 0
                self.c.usage = 0
                self.c.cum_evaluated = 0
                self.c.cum_correct = 0
                self._c_temp = None

                if new_distillator is not None:
                    self._active_distillator = new_distillator
                else:
                    self._active_distillator = self._create_distillator()
                    if td is not None and len(td) > 0:
                        self._active_distillator.prototypes = td

                if self.W is not None:
                    for wx, wy in self.W:
                        self._active_distillator.learn_one(self._normalize_x(wx), wy)
                    if hasattr(self._active_distillator, "train_from_buffer"):
                        self._active_distillator.train_from_buffer()

                self.W = None
                self._clear_window_results()
                self.b = None
            else:
                self._log("FALSE ALARM ON DRIFT SIGNAL - RETAINING WARNING WINDOW")
        else:
            self._log("FALSE ALARM ON DRIFT SIGNAL")
            self._log("EARLY_DRIFT_1 - NOT ENOUGH EXAMPLES DURING WARNING")
            self.last_drift_was_early = True
            self.last_drift_was_false_alarm = True
            self.last_drift_was_inserted = False
            if self.save_classifiers_on_false_alarm and self._c_temp is not None:
                self._insert_into_ch(check_error_improvement=False)

        self.last_drift_at = self.n_samples_seen
        if self._drift_detector is self._warning_detector:
            self._drift_detector = self.drift_detector.clone()
            self._warning_detector = self._drift_detector
        else:
            self._drift_detector = self.drift_detector.clone()
            self._warning_detector = self._warning_detector.clone()
        self._prev_warning_zone = False

    def _update_warning_and_topology_after_drift(
        self, x: dict, y: base.typing.ClfTarget, is_warning_zone: bool
    ) -> None:
        reset_by_timeout_or_eoz = False
        if self.W is not None:
            timeout = len(self.W) > self.mu
            end_of_zone = self.end_of_warning_after_warning_zone and not is_warning_zone
            if timeout or end_of_zone:
                self._log("WARNING WINDOW RESET (Timeout or end of warning zone)")
                self._reset_warning_window(starting_ww=True)
                reset_by_timeout_or_eoz = True

        if self._train_active_on_warn:
            self.c.learn_one(x, y)
        if self.W is not None:
            if self.train_on_warn and self.b is not None:
                self.b.learn_one(x, y)
            self.W.append((dict(x), y))
        elif not reset_by_timeout_or_eoz:
            if not self._train_active_on_warn:
                self.c.learn_one(x, y)
            self._active_distillator.learn_one(self._normalize_x(x), y)

    # ------------------------------------------------------------------
    # Main River Classifier API
    # ------------------------------------------------------------------

    def learn_one(self, x: dict, y: base.typing.ClfTarget) -> None:
        self.n_samples_seen += 1
        self.last_drift_was_false_alarm = False
        is_on_warning = self.W is not None

        # 1. Pre-training evaluation during active warning window
        self._evaluate_warning_window(x, y)

        # 2. Optional pre-drift training of active classifier
        if not self.train_after_drift_evaluation:
            if self._train_active_on_warn or not is_on_warning:
                self.c.learn_one(x, y)

        y_pred = self.c.predict_one(x)
        drift_input = int(str(y) != str(y_pred))
        self.c.cum_evaluated += 1
        self.c.cum_correct += 1 - drift_input

        if self.W is not None and self.b is not None:
            self._update_weights(x, y)

        if not self.train_after_drift_evaluation:
            if self.W is not None:
                timeout = len(self.W) > self.mu
                end_of_zone = self.end_of_warning_after_warning_zone and not getattr(
                    self._warning_detector, "warning_detected", False
                )
                if timeout or end_of_zone:
                    self._log("WARNING WINDOW RESET (Timeout or end of warning zone)")
                    self._reset_warning_window(starting_ww=True)
                else:
                    if self.train_on_warn and self.b is not None:
                        self.b.learn_one(x, y)
                    self.W.append((dict(x), y))
            else:
                self._active_distillator.learn_one(self._normalize_x(x), y)

        # 3. Update drift and warning detectors
        was_warning_zone = getattr(self._warning_detector, "warning_detected", False)
        self._drift_detector.update(drift_input)
        if self._warning_detector is not self._drift_detector:
            self._warning_detector.update(drift_input)

        is_warning_zone = getattr(self._warning_detector, "warning_detected", False)
        if hasattr(self._warning_detector, "is_warning_edge"):
            is_warn_edge = bool(self._warning_detector.is_warning_edge)
        else:
            is_warn_edge = (not was_warning_zone) and is_warning_zone

        # 4. Warning edge & drift transitions
        if is_warn_edge:
            self._start_warning_window()

        if self._drift_detector.drift_detected:
            self._handle_drift(x, y)

        # 5. Optional post-drift training
        if self.train_after_drift_evaluation:
            self._update_warning_and_topology_after_drift(x, y, is_warning_zone)

        self._prev_warning_zone = is_warning_zone

    def _active_predictor(self) -> LocalLearner:
        if self.W is not None and len(self.W) >= self.theta and self.b is not None:
            if self.weighted_tests_on_warn and self.b_weight > self.c_weight:
                self._log("Predicting with BACKGROUND model")
                return self.b
            if not self.weighted_tests_on_warn and self.test_with_bkg_on_warn:
                return self.b
        return self.c

    def predict_one(self, x: dict, **kwargs: typing.Any) -> base.typing.ClfTarget | None:
        return self._active_predictor().predict_one(x)

    def predict_proba_one(
        self, x: dict, **kwargs: typing.Any
    ) -> dict[base.typing.ClfTarget, float]:
        return self._active_predictor().predict_proba_one(x)

    # ------------------------------------------------------------------
    # Concept History similarity, retrieval, eviction & selection
    # ------------------------------------------------------------------

    def _distance(self, p1: dict, p2: dict) -> float:
        if self.custom_distance_func is not None:
            return float(self.custom_distance_func(p1, p2))
        if self.dist_metric.lower() == "mahalanobis" and self._inv_cov is not None:
            keys = self._feature_keys if self._feature_keys is not None else sorted(p1.keys())
            dist = mahalanobis_distance(p1, p2, self._inv_cov, features=keys)
            return float(np.sqrt(self._det_root) * dist)
        return utils.math.minkowski_distance(p1, p2, 2)

    def _to_2d_array(self, p_set) -> np.ndarray:
        if isinstance(p_set, TopologyDistillator):
            return p_set.prototypes_array
        if isinstance(p_set, np.ndarray):
            return p_set
        if not p_set:
            return np.zeros((0, 0), dtype=np.float64)
        include_target = getattr(self._active_distillator, "include_target", False)
        first = p_set[0]
        if isinstance(first, tuple):
            keys = (
                self.feature_subset
                if self.feature_subset is not None
                else (
                    self._feature_keys
                    if self._feature_keys is not None
                    else sorted(first[0].keys())
                )
            )
            if include_target:
                rows = [
                    [float(x.get(k, 0.0)) for k in keys] + [float(y) if y is not None else 0.0]
                    for x, y in p_set
                ]
            else:
                rows = [[float(x.get(k, 0.0)) for k in keys] for x, _ in p_set]
            return np.array(rows, dtype=np.float64)
        if isinstance(first, dict):
            keys = (
                self.feature_subset
                if self.feature_subset is not None
                else (
                    self._feature_keys if self._feature_keys is not None else sorted(first.keys())
                )
            )
            if include_target and "label" in first:
                rows = [
                    [float(d.get(k, 0.0)) for k in keys] + [float(d.get("label", 0.0))]
                    for d in p_set
                ]
            else:
                rows = [[float(d.get(k, 0.0)) for k in keys] for d in p_set]
            return np.array(rows, dtype=np.float64)
        return np.asarray(p_set, dtype=np.float64)

    def _mean_nn_dist(self, p_set, pg_set) -> float:
        if self.custom_mean_nn_dist_func is not None:
            mean_d = float(self.custom_mean_nn_dist_func(p_set, pg_set))
            self._log(
                f"    _mean_nn_dist (custom): len(P)={len(p_set)}, len(PG)={len(pg_set)}, mean={mean_d:.4f}"
            )
            return mean_d
        if self.custom_distance_func is not None:
            dists = [
                min(self._distance(p[0] if isinstance(p, tuple) else p, pg) for pg in pg_set)
                for p in p_set
            ]
            return sum(dists) / len(dists)

        A = self._to_2d_array(p_set)
        B = self._to_2d_array(pg_set)
        if len(A) == 0 or len(B) == 0:
            return float("inf")

        diff = A[:, None, :] - B[None, :, :]
        if self.dist_metric.lower() == "euclidean" or self._inv_cov is None:
            dist_sq = np.sum(diff**2, axis=2)
        else:
            temp = np.einsum("ijk,kl->ijl", diff, self._inv_cov)
            dist_sq = np.einsum("ijl,ijl->ij", temp, diff)

        dist = np.sqrt(np.maximum(dist_sq, 0.0))
        if self._det_root != 1.0:
            dist = np.sqrt(self._det_root) * dist
        mean_d = float(np.mean(np.min(dist, axis=1)))
        self._log(f"    _mean_nn_dist: len(P)={len(A)}, len(PG)={len(B)}, mean={mean_d:.4f}")
        return mean_d

    def _group_prototypes_for_match(self, group: dict):
        if self.custom_mean_nn_dist_func is None and group.get("distillator") is not None:
            return group["distillator"]
        return group["prototypes"]

    def _find_group(self, p_set) -> dict | None:
        if not self.CH or p_set is None or len(p_set) == 0:
            return None
        min_dist = float("inf")
        best_group = None
        for i, group in enumerate(self.CH):
            dist = self._mean_nn_dist(p_set, self._group_prototypes_for_match(group))
            self._log(f"  Group {i} distance: {dist:.4f}", level=2)
            if dist < min_dist:
                min_dist = dist
                best_group = group

        matched_gid = (
            best_group.get("group_id")
            if (best_group is not None and min_dist < self.sigma)
            else None
        )
        self._log(
            f"FindGroup: min_dist={min_dist:.4f} (threshold sigma={self.sigma}) -> Matched Group: {matched_gid}",
            level=1,
        )
        return best_group if (best_group is not None and min_dist < self.sigma) else None

    def _find_groups(self, p_set) -> list[dict]:
        if not self.CH or p_set is None or len(p_set) == 0:
            return []
        matched = []
        for i, group in enumerate(self.CH):
            dist = self._mean_nn_dist(p_set, self._group_prototypes_for_match(group))
            self._log(f"  Group {i} distance: {dist:.4f}", level=2)
            if dist < self.sigma:
                matched.append((dist, group))
        matched.sort(key=lambda item: item[0])
        self._log(
            f"FindGroups (mc=True): matched {len(matched)} groups with dist < sigma={self.sigma}",
            level=1,
        )
        return [g for _, g in matched]

    def _rank_concept_history_groups(self, candidate_groups: list[dict]) -> dict | None:
        min_err = float("inf")
        best_group = None
        for group in candidate_groups:
            for learner in group.get("classifiers", []):
                err = self._window_error(learner)
                if not np.isnan(err):
                    self._log(
                        f"  Candidate from group {group.get('group_id')} has err={err:.4f}", level=2
                    )
                    if err < min_err:
                        self._log(
                            f"  Candidate from group {group.get('group_id')} improves min with err={err:.4f}",
                            level=2,
                        )
                        min_err = err
                        best_group = group
        return (
            best_group
            if best_group is not None
            else (candidate_groups[0] if candidate_groups else None)
        )

    def _remove_classifier(self, group: dict) -> None:
        classifiers = group["classifiers"]
        if not classifiers:
            return
        g_id = group.get("group_id")
        policy = self.group_replacement_policy.lower().replace("_", " ").replace("-", " ").strip()
        if policy in ("fifo", "first in first out"):
            rem_idx = min(range(len(classifiers)), key=lambda i: classifiers[i].last_time_used)
        elif policy in ("least used model", "least used"):
            min_usage = min(m.usage for m in classifiers)
            candidates = [i for i, m in enumerate(classifiers) if m.usage == min_usage]
            rem_idx = min(candidates, key=lambda i: classifiers[i].last_time_used)
        else:
            raise ValueError(
                f"Unknown group_replacement_policy: '{self.group_replacement_policy}'. "
                "Expected 'fifo' or 'least used model'."
            )

        removed = classifiers.pop(rem_idx)
        self._log(
            f"RemoveClassifier in Group {g_id} (Policy='{policy}'): evicted idx={rem_idx} "
            f"(c_at={removed.created_at}, ltu={removed.last_time_used}, usage={removed.usage}). "
            f"Size: {len(classifiers) + 1} -> {len(classifiers)}",
            level=1,
        )
        if self.log_level >= 3:
            self._log(f"    Evicted model structure: {self._describe_model(removed)}", level=3)

    def _select_drift(
        self,
        c: LocalLearner,
        b: LocalLearner,
        gh_models: list[LocalLearner],
        gh_prototypes,
    ) -> tuple[LocalLearner, typing.Any, bool, int | None]:
        err_c = self._window_error(c, fallback_cum=self.active_better_when_bkg_nan)
        err_b = self._window_error(b, fallback_cum=False)

        if self.active_better_when_bkg_nan:
            active_improves_bkg = (not np.isnan(err_c)) and (np.isnan(err_b) or err_c <= err_b)
        else:
            active_improves_bkg = (
                (not np.isnan(err_c)) and (not np.isnan(err_b)) and (err_c <= err_b)
            )

        has_recurring = False
        best_idx: int | None = None
        best_gh_err = float("nan")

        if gh_models:
            gh_errs = [self._window_error(m) for m in gh_models]
            valid = [i for i, err in enumerate(gh_errs) if not np.isnan(err)]
            if valid:
                min_gh_err = min(gh_errs[i] for i in valid)
                tied = [i for i in valid if gh_errs[i] == min_gh_err]
                if len(tied) == 1:
                    best_idx = tied[0]
                elif self.priority_of_recurring.lower() == "old":
                    best_idx = min(tied, key=lambda i: gh_models[i].last_time_used)
                else:
                    best_idx = max(tied, key=lambda i: gh_models[i].last_time_used)

                has_recurring = True
                best_gh_err = gh_errs[best_idx]

                if self.log_level >= 2:
                    cand_info = [
                        f"#{i}(err={gh_errs[i]:.4f}, c_at={m.created_at}, ltu={m.last_time_used}, usage={m.usage})"
                        for i, m in enumerate(gh_models)
                    ]
                    self._log(f"SelectDrift candidates evaluation: {', '.join(cand_info)}", level=2)
                    self._log(
                        f"  err_c={err_c:.4f}, err_b={err_b:.4f}, min_gh_err={min_gh_err:.4f}",
                        level=2,
                    )

        if active_improves_bkg:
            active_improves_recurring = (
                (not np.isnan(err_c)) and (not np.isnan(best_gh_err)) and (err_c <= best_gh_err)
                if has_recurring
                else True
            )
            if has_recurring and not active_improves_recurring and best_idx is not None:
                self._log(
                    f"SelectDrift: TRANSITION TO HISTORICAL MODEL (selected #{best_idx}, err={best_gh_err:.4f})",
                    level=1,
                )
                return gh_models[best_idx].clone(), gh_prototypes, False, best_idx
            self._log(
                f"SelectDrift: FALSE ALARM - Retaining Foreground (err_c={err_c:.4f} <= err_b={err_b:.4f})",
                level=1,
            )
            return c, [], True, None

        ch_improves_bkg = (
            has_recurring
            and best_idx is not None
            and (not np.isnan(best_gh_err))
            and (not np.isnan(err_b))
            and (best_gh_err <= err_b)
        )
        if ch_improves_bkg and best_idx is not None:
            self._log(
                f"SelectDrift: TRANSITION TO HISTORICAL MODEL (selected #{best_idx}, err={best_gh_err:.4f})",
                level=1,
            )
            return gh_models[best_idx].clone(), gh_prototypes, False, best_idx

        self._log("SelectDrift: TRANSITION TO BACKGROUND MODEL", level=1)
        return b.clone(), [], False, None
