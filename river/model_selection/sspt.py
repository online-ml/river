import collections
import copy
import math
import numbers
import random
import typing

import numpy as np

# TODO use lazy imports where needed
from river import anomaly, base, compose, drift, metrics, utils

ModelWrapper = collections.namedtuple("ModelWrapper", "estimator metric")


class SSPT(base.Estimator):
    """Single-pass Self Parameter Tuning.

    Parameters
    ----------
    estimator
        The estimator whose hyperparameters are going to be tuned.
    metric
        Evaluation metric compatible with the underlying learning task. It is
        used to select the best learning models among the set of candidates.
    params_range
    drift_input
        Define how the ground truth and the predicted labels are combined to be
        passed to the concept drift detector. For example, in regression tasks is usual
        to take the difference between $y$ and $\\hat{y}$, whereas in classification
        a 0-1 loss is commonly used.
    grace_period
        Define the interval before SSPT checks for convergence and updates the candidate
        models hyperparameters.
    drift_detector
    convergence_sphere

    seed
        Random seed for reproducibility.

    Examples
    --------

    >>> import numbers

    >>> from river import compose
    >>> from river.datasets import synth
    >>> from river import drift
    >>> from river import metrics
    >>> from river import model_selection
    >>> from river import preprocessing
    >>> from river import tree

    >>> dataset = synth.Mv(seed=8).take(1000)

    >>> # Dealing with mixed variable types
    >>> num = compose.SelectType(numbers.Number) | preprocessing.StandardScaler()
    >>> cat = compose.SelectType(str) | preprocessing.OneHotEncoder()

    Let's compare a tree using its default hyperparameters versus one realization of the
    same tree model where the hyperparameters are selected automatically.

    >>> # Vanilla HTR
    >>> htr = (num + cat) | tree.HoeffdingTreeRegressor()

    >>> # Self-tuning model
    >>> sspt = model_selection.SSPT(
    ...     estimator=(num + cat) | tree.HoeffdingTreeRegressor(),
    ...     metric=metrics.MAE(),
    ...     grace_period=100,
    ...     params_range={
    ...         "HoeffdingTreeRegressor": {
    ...             "tau": (float, (0.01, 0.09)),
    ...             "delta": (float, (0.00001, 0.001)),
    ...             "grace_period": (int, (50, 200)),
    ...         },
    ...     },
    ...     drift_detector=drift.ADWIN(),
    ...     drift_input=lambda yt, yp: yt - yp,
    ...     seed=42,
    ... )

    >>> # We will measure the same metrics for both the contenders
    >>> m_htr = metrics.MAE() + metrics.R2()
    >>> m_sspt = metrics.MAE() + metrics.R2()

    >>> for x, y in dataset:
    ...     yp = htr.predict_one(x)
    ...     yp_t = sspt.predict_one(x)
    ...
    ...     m_htr.update(y, yp)
    ...     m_sspt.update(y, yp_t)
    ...
    ...     htr.learn_one(x, y)
    ...     sspt.learn_one(x, y)

    Now we check the performance of the standalone tree:

    >>> m_htr
        MAE: 11.830555
        R2: 0.712767

    >>> m_sspt
        MAE: 9.782435
        R2: 0.783856

    References
    ----------
    [1]: Veloso, B., Gama, J., Malheiro, B., & Vinagre, J. (2021). Hyperparameter self-tuning
    for data streams. Information Fusion, 76, 75-86.

    """

    _START_RANDOM = "random"
    _START_WARM = "warm"

    def __init__(
        self,
        estimator: base.Estimator,
        metric: metrics.base.Metric,
        params_range: typing.Dict[str, typing.Tuple],
        drift_input: typing.Callable[[float, float], float],
        grace_period: int = 500,
        drift_detector: base.DriftDetector = drift.ADWIN(),
        convergence_sphere: float = 0.001,
        seed: int = None,
    ):
        super().__init__()
        self.estimator = estimator
        self.metric = metric
        self.params_range = params_range
        self.drift_input = drift_input

        self.grace_period = grace_period
        self.drift_detector = drift_detector
        self.convergence_sphere = convergence_sphere

        self.seed = seed

        self._n = 0
        self._converged = False
        self._rng = random.Random(self.seed)

        self._best_estimator = None
        self._simplex = self._create_simplex(estimator)

        # Models expanded from the simplex
        self._expanded: typing.Optional[typing.Dict] = None

        # Convergence criterion
        self._old_centroid = None

        # Meta-programming
        border = self.estimator
        if isinstance(border, compose.Pipeline):
            border = border[-1]

        if isinstance(border, (base.Classifier, base.Regressor)):
            self._scorer_name = "predict_one"
        elif isinstance(border, anomaly.base.AnomalyDetector):
            self._scorer_name = "score_one"
        elif isinstance(border, anomaly.base.AnomalyDetector):
            self._scorer_name = "classify"

    def __generate(self, hp_data) -> numbers.Number:
        hp_type, hp_range = hp_data
        if hp_type == int:
            return self._rng.randint(hp_range[0], hp_range[1])
        elif hp_type == float:
            return self._rng.uniform(hp_range[0], hp_range[1])

    def __combine(self, hp_data, hp_est_1, hp_est_2, func) -> numbers.Number:
        hp_type, hp_range = hp_data
        new_val = func(hp_est_1, hp_est_2)

        # Range sanity checks
        if new_val < hp_range[0]:
            new_val = hp_range[0]
        if new_val > hp_range[1]:
            new_val = hp_range[1]

        new_val = round(new_val, 0) if hp_type == int else new_val
        return new_val

    def __flatten(self, prefix, scaled_hps, hp_data, est_data):
        hp_range = hp_data[1]
        interval = hp_range[1] - hp_range[0]
        scaled_hps[prefix] = (est_data - hp_range[0]) / interval

    def _traverse_hps(
        self, operation: str, hp_data: dict, est_1, *, func=None, est_2=None, hp_prefix=None, scaled_hps=None
    ) -> typing.Optional[typing.Union[dict, numbers.Number]]:
        """Traverse the hyperparameters of the estimator/pipeline and perform an operation.

        Parameters
        ----------
        operation
            The operation that is intented to apply over the hyperparameters. Can be either:
            "combine" (combine parameters from two pipelines), "scale" (scale a flattened
            version of the hyperparameter hierarchy to use in the stopping criteria), or
            "generate" (create a new hyperparameter set candidate).
        hp_data
            The hyperparameter data which was passed by the user. Defines the ranges to
            explore for each hyperparameter.
        est_1
            The hyperparameter structure of the first pipeline/estimator. Such structure is obtained
            via a `_get_params()` method call. Both 'hp_data' and 'est_1' will be jointly traversed.
        func
            A function that is used to combine the values in `est_1` and `est_2`, in case
            `operation="combine"`.
        est_2
            A second pipeline/estimator which is going to be combined with `est_1`, in case
            `operation="combine"`.
        hp_prefix
            A hyperparameter prefix which is used to identify each hyperparameter in the hyperparameter
            hierarchy when `operation="scale"`. The recursive traversal will modify this prefix accordingly
            to the current position in the hierarchy. Initially it is set to `None`.
        scaled_hps
            Flattened version of the hyperparameter hierarchy which is used for evaluating stopping criteria.
            Set to `None` and defined automatically when `operation="scale"`.
        """

        # Sub-component needs to be instantiated
        if isinstance(est_1, tuple):
            sub_class, est_1 = est_1

            if operation == "combine":
                est_2 = est_2[1]
            else:
                est_2 = {}

            sub_config = {}
            for sub_hp_name, sub_hp_data in hp_data.items():
                if operation == "scale":
                    sub_hp_prefix = hp_prefix + "__" + sub_hp_name
                else:
                    sub_hp_prefix = None

                sub_config[sub_hp_name] = self._traverse_hps(
                    operation=operation,
                    hp_data=sub_hp_data,
                    est_1=est_1[sub_hp_name],
                    func=func,
                    est_2=est_2.get(sub_hp_name, None),
                    hp_prefix=sub_hp_prefix,
                    scaled_hps=scaled_hps,
                )
            return sub_class(**sub_config)

        # We reached the numeric parameters
        if isinstance(est_1, numbers.Number):
            if operation == "generate":
                return self.__generate(hp_data)
            if operation == "scale":
                self.__flatten(hp_prefix, scaled_hps, hp_data, est_1)
                return
            # combine
            return self.__combine(hp_data, est_1, est_2, func)

        # The sub-parameters need to be expanded
        config = {}
        for sub_hp_name, sub_hp_data in hp_data.items():
            sub_est_1 = est_1[sub_hp_name]

            if operation == "combine":
                sub_est_2 = est_2[sub_hp_name]
            else:
                sub_est_2 = {}

            if operation == "scale":
                sub_hp_prefix = hp_prefix + "__" + sub_hp_name if len(hp_prefix) > 0 else sub_hp_name
            else:
                sub_hp_prefix = None

            config[sub_hp_name] = self._traverse_hps(
                operation=operation,
                hp_data=sub_hp_data,
                est_1=sub_est_1,
                func=func,
                est_2=sub_est_2,
                hp_prefix=sub_hp_prefix,
                scaled_hps=scaled_hps,
            )

        return config

    def _random_config(self):
        return self._traverse_hps(
            operation="generate",
            hp_data=self.params_range,
            est_1=self.estimator._get_params()
        )

    def _create_simplex(self, model) -> typing.List:
        # The simplex is divided in:
        # * 0: the best model
        # * 1: the 'good' model
        # * 2: the worst model
        simplex = [None] * 3

        simplex[0] = ModelWrapper(
            self.estimator.clone(self._random_config(), include_attributes=True),
            self.metric.clone(include_attributes=True),
        )
        simplex[1] = ModelWrapper(
            model.clone(self._random_config(), include_attributes=True),
            self.metric.clone(include_attributes=True),
        )
        simplex[2] = ModelWrapper(
            self.estimator.clone(self._random_config(), include_attributes=True),
            self.metric.clone(include_attributes=True),
        )

        return simplex

    def _sort_simplex(self):
        """Ensure the simplex models are ordered by predictive performance."""
        if self.metric.bigger_is_better:
            self._simplex.sort(key=lambda mw: mw.metric.get(), reverse=True)
        else:
            self._simplex.sort(key=lambda mw: mw.metric.get())

    def _gen_new_estimator(self, e1, e2, func):
        """Generate new configuration given two estimators and a combination function."""

        est_1_hps = e1.estimator._get_params()
        est_2_hps = e2.estimator._get_params()

        new_config = self._traverse_hps(
            operation="combine",
            hp_data=self.params_range,
            est_1=est_1_hps,
            func=func,
            est_2=est_2_hps
        )

        # Modify the current best contender with the new hyperparameter values
        new = ModelWrapper(
            copy.deepcopy(self._simplex[0].estimator),
            self.metric.clone(include_attributes=True),
        )
        new.estimator.mutate(new_config)

        return new

    def _nelder_mead_expansion(self) -> typing.Dict:
        """Create expanded models given the simplex models."""
        expanded = {}
        # Midpoint between 'best' and 'good'
        expanded["midpoint"] = self._gen_new_estimator(
            self._simplex[0], self._simplex[1], lambda h1, h2: (h1 + h2) / 2
        )

        # Reflection of 'midpoint' towards 'worst'
        expanded["reflection"] = self._gen_new_estimator(
            expanded["midpoint"], self._simplex[2], lambda h1, h2: 2 * h1 - h2
        )
        # Expand the 'reflection' point
        expanded["expansion"] = self._gen_new_estimator(
            expanded["reflection"], expanded["midpoint"], lambda h1, h2: 2 * h1 - h2
        )
        # Shrink 'best' and 'worst'
        expanded["shrink"] = self._gen_new_estimator(
            self._simplex[0], self._simplex[2], lambda h1, h2: (h1 + h2) / 2
        )
        # Contraction of 'midpoint' and 'worst'
        expanded["contraction1"] = self._gen_new_estimator(
            expanded["midpoint"], self._simplex[2], lambda h1, h2: (h1 + h2) / 2
        )
        # Contraction of 'midpoint' and 'reflection'
        expanded["contraction2"] = self._gen_new_estimator(
            expanded["midpoint"], expanded["reflection"], lambda h1, h2: (h1 + h2) / 2
        )

        return expanded

    def _nelder_mead_operators(self):
        b = self._simplex[0].metric
        g = self._simplex[1].metric
        w = self._simplex[2].metric
        r = self._expanded["reflection"].metric
        c1 = self._expanded["contraction1"].metric
        c2 = self._expanded["contraction2"].metric

        if c1.is_better_than(c2):
            self._expanded["contraction"] = self._expanded["contraction1"]
        else:
            self._expanded["contraction"] = self._expanded["contraction2"]
        if r.is_better_than(g):
            if b.is_better_than(r):
                self._simplex[2] = self._expanded["reflection"]
            else:
                e = self._expanded["expansion"].metric
                if e.is_better_than(b):
                    self._simplex[2] = self._expanded["expansion"]
                else:
                    self._simplex[2] = self._expanded["reflection"]
        else:
            if r.is_better_than(w):
                self._simplex[2] = self._expanded["reflection"]
            else:
                c = self._expanded["contraction"].metric
                if c.is_better_than(w):
                    self._simplex[2] = self._expanded["contraction"]
                else:
                    self._simplex[2] = self._expanded["shrink"]
                    self._simplex[1] = self._expanded["midpoint"]

        self._sort_simplex()

    def _normalize_flattened_hyperspace(self, orig):
        scaled = {}
        self._traverse_hps(
            operation="scale",
            hp_data=self.params_range,
            est_1=orig,
            hp_prefix="",
            scaled_hps=scaled
        )
        return scaled

    @property
    def _models_converged(self) -> bool:
        # Normalize params to ensure they contribute equally to the stopping criterion

        # 1. Simplex in sphere
        scaled_params_b = self._normalize_flattened_hyperspace(
            self._simplex[0].estimator._get_params()
        )
        scaled_params_g = self._normalize_flattened_hyperspace(
            self._simplex[1].estimator._get_params()
        )
        scaled_params_w = self._normalize_flattened_hyperspace(
            self._simplex[2].estimator._get_params()
        )

        max_dist = max(
            [
                utils.math.minkowski_distance(scaled_params_b, scaled_params_g, p=2),
                utils.math.minkowski_distance(scaled_params_b, scaled_params_w, p=2),
                utils.math.minkowski_distance(scaled_params_g, scaled_params_w, p=2),
            ]
        )

        hyper_points = [
            list(scaled_params_b.values()),
            list(scaled_params_g.values()),
            list(scaled_params_w.values()),
        ]

        vectors = np.array(hyper_points)
        new_centroid = dict(zip(scaled_params_b.keys(), np.mean(vectors, axis=0)))
        centroid_distance = utils.math.minkowski_distance(
            self._old_centroid, new_centroid, p=2
        )
        self._old_centroid = new_centroid
        ndim = len(scaled_params_b)
        r_sphere = max_dist * math.sqrt((ndim / (2 * (ndim + 1))))

        if r_sphere < self.convergence_sphere or centroid_distance == 0:
            return True

        return False

    def _learn_converged(self, x, y):
        scorer = getattr(self._best_estimator, self._scorer_name)
        y_pred = scorer(x)

        input = self.drift_input(y, y_pred)
        self.drift_detector.update(input)

        # We need to start the optimization process from scratch
        if self.drift_detector.drift_detected:
            self._n = 0
            self._converged = False
            self._simplex = self._create_simplex(self._best_estimator)

            # There is no proven best model right now
            self._best_estimator = None
            return

        self._best_estimator.learn_one(x, y)

    def _learn_not_converged(self, x, y):
        for wrap in self._simplex:
            scorer = getattr(wrap.estimator, self._scorer_name)
            y_pred = scorer(x)
            wrap.metric.update(y, y_pred)
            wrap.estimator.learn_one(x, y)

        # Keep the simplex ordered
        self._sort_simplex()

        if not self._expanded:
            self._expanded = self._nelder_mead_expansion()

        for wrap in self._expanded.values():
            scorer = getattr(wrap.estimator, self._scorer_name)
            y_pred = scorer(x)
            wrap.metric.update(y, y_pred)
            wrap.estimator.learn_one(x, y)

        if self._n == self.grace_period:
            self._n = 0
            # 1. Simplex in sphere
            scaled_params_b = self._normalize_flattened_hyperspace(
                self._simplex[0].estimator._get_params(),
            )
            scaled_params_g = self._normalize_flattened_hyperspace(
                self._simplex[1].estimator._get_params(),
            )
            scaled_params_w = self._normalize_flattened_hyperspace(
                self._simplex[2].estimator._get_params(),
            )
            hyper_points = [
                list(scaled_params_b.values()),
                list(scaled_params_g.values()),
                list(scaled_params_w.values()),
            ]
            vectors = np.array(hyper_points)
            self._old_centroid = dict(
                zip(scaled_params_b.keys(), np.mean(vectors, axis=0))
            )

            # Update the simplex models using Nelder-Mead heuristics
            self._nelder_mead_operators()

            # Discard expanded models
            self._expanded = None

            if self._models_converged:
                self._converged = True
                self._best_estimator = self._simplex[0].estimator

    def learn_one(self, x, y):
        self._n += 1

        if self.converged:
            self._learn_converged(x, y)
        else:
            self._learn_not_converged(x, y)

    @property
    def best(self):
        if not self._converged:
            # Lazy selection of the best model
            self._sort_simplex()
            return self._simplex[0].estimator

        return self._best_estimator

    @property
    def converged(self):
        return self._converged

    def predict_one(self, x, **kwargs):
        try:
            return self.best.predict_one(x, **kwargs)
        except NotImplementedError:
            border = self.best
            if isinstance(border, compose.Pipeline):
                border = border[-1]
            raise AttributeError(
                f"'predict_one' is not supported in {border.__class__.__name__}."
            )

    def predict_proba_one(self, x, **kwargs):
        try:
            return self.best.predict_proba_one(x, **kwargs)
        except NotImplementedError:
            border = self.best
            if isinstance(border, compose.Pipeline):
                border = border[-1]
            raise AttributeError(
                f"'predict_proba_one' is not supported in {border.__class__.__name__}."
            )

    def score_one(self, x, **kwargs):
        try:
            return self.best.score_one(x, **kwargs)
        except NotImplementedError:
            border = self.best
            if isinstance(border, compose.Pipeline):
                border = border[-1]
            raise AttributeError(
                f"'score_one' is not supported in {border.__class__.__name__}."
            )

    def debug_one(self, x, **kwargs):
        try:
            return self.best.score_one(x, **kwargs)
        except NotImplementedError:
            raise AttributeError(
                f"'debug_one' is not supported in {self.best.__class__.__name__}."
            )
