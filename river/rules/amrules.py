from __future__ import annotations

import collections
import functools
import io
import math
import typing

from river import anomaly, base, drift, linear_model, stats
from river.tree import split_criterion
from river.tree import splitter as spl

from .base import HoeffdingRule


class MeanRegressor(base.Regressor):
    def __init__(self):
        self.mean = stats.Mean()

    def learn_one(self, x: dict, y: base.typing.RegTarget, w: int = 1):
        self.mean.update(y, w)
        return self

    def predict_one(self, x: dict):
        return self.mean.get()


class AdaptiveRegressor(base.Regressor):
    """This predictor selects between the target mean and the the user-selected regression model
     automatically. Faded error metrics are kept for both the predictors, and the most
    accurate one is selected automatically for each incoming instance.
    """

    def __init__(self, model_predictor: base.Regressor, fading_factor: float):
        self.model_predictor = model_predictor
        self.mean_predictor = MeanRegressor()
        self.fading_factor = fading_factor

        self._mae_mean = 0.0
        self._mae_model = 0.0

    def learn_one(self, x: dict, y: base.typing.RegTarget, w: int = 1):
        abs_error_mean = abs(y - self.mean_predictor.predict_one(x))  # type: ignore
        abs_error_model = abs(y - self.model_predictor.predict_one(x))  # type: ignore

        self._mae_mean = self.fading_factor * self._mae_mean + abs_error_mean
        self._mae_model = self.fading_factor * self._mae_model + abs_error_model

        self.mean_predictor.learn_one(x, y, w)

        for _ in range(int(w)):
            self.model_predictor.learn_one(x, y)

        return self

    def predict_one(self, x: dict):
        if self._mae_mean <= self._mae_model:
            return self.mean_predictor.predict_one(x)
        else:
            return self.model_predictor.predict_one(x)


class RegRule(HoeffdingRule, base.Regressor, anomaly.base.AnomalyDetector):
    def __init__(
        self,
        template_splitter,
        split_criterion,
        pred_model,
        drift_detector,
    ):
        super().__init__(
            template_splitter=template_splitter,
            split_criterion=split_criterion,
        )
        self.pred_model = pred_model
        self.drift_detector = drift_detector

        self._target_stats = stats.Var()
        self._feat_stats = collections.defaultdict(functools.partial(stats.Var))

    def new_nominal_splitter(self):
        return spl.NominalSplitterReg()

    @property
    def statistics(self):
        return self._target_stats

    @statistics.setter
    def statistics(self, target_stats):
        self._target_stats = target_stats

    def _update_target_stats(self, y, w):
        self._target_stats.update(y, w)

    def _update_feature_stats(self, feat_name, feat_val, w):
        if feat_name not in self.nominal_features:
            self._feat_stats[feat_name].update(feat_val, w)

    def drift_test(self, y, y_pred):
        abs_error = abs(y - y_pred)
        self.drift_detector.update(abs_error)

        return self.drift_detector.drift_detected

    def score_one(self, x) -> float:
        """Rule anomaly score.

        The more negative the score, the more anomalous is the instance regarding the subspace
        covered by the rule. Instances whose score is greater than zero are considered normal.
        Different threshold values can be used to decide upon discarding or using the instance
        for training. A small negative threshold is considered to be a conservative choice,
        whereas cutting at zero is a more aggressive choice.

        Parameters
        ----------
        x
            A dictionary of features.

        Returns
        -------
        An anomaly score. The more negative the score, the more anomalous is the instance.


        """
        score = 0.0
        hits = 0

        for feat_name, feat_val in x.items():
            # Skip nominal features
            if feat_name in self.nominal_features:
                continue

            mean = self._feat_stats[feat_name].mean.get()
            var = self._feat_stats[feat_name].get()

            if var > 0:
                # One tailed variant of Chebyshev's inequality: ported from the MOA code
                proba = (2 * var) / (var + (feat_val - mean) ** 2)

                if 0 < proba < 1:
                    score += math.log(proba) - math.log(1 - proba)
                    hits += 1

        return score / hits if hits > 0 else 0.0

    def learn_one(self, x: dict, y: base.typing.RegTarget, w: int = 1):  # type: ignore
        self.update(x, y, w)
        self.pred_model.learn_one(x, y, w)

        return self

    def predict_one(self, x: dict):
        return self.pred_model.predict_one(x)


class AMRules(base.Regressor):
    """Adaptive Model Rules.

    AMRules[^1] is a rule-based algorithm for incremental regression tasks. AMRules relies on the
    Hoeffding bound to build its rule set, similarly to Hoeffding Trees. The Variance-Ratio
    heuristic is used to evaluate rules' splits. Moreover, this rule-based regressor has
    additional capacities not usually found in decision trees.

    Firstly, each created decision rule has a built-in drift detection mechanism. Every time a
    drift is detected, the affected decision rule is removed. In addition, AMRules' rules also
    have anomaly detection capabilities. After a warm-up period, each rule tests whether or not
    the incoming instances are anomalies. Anomalous instances are not used for training.

    Every time no rule is covering an incoming example, a default rule is used to learn
    from it. A rule covers an instance when all of the rule's literals (tests joined by the
    logical operation `and`) match the input case. The default rule is also applied for predicting
    examples not covered by any rules from the rule set.

    Parameters
    ----------
    n_min
        The total weight that must be observed by a rule between expansion attempts.
    delta
        The split test significance. The split confidence is given by `1 - delta`.
    tau
        The tie-breaking threshold.
    pred_type
        The prediction strategy used by the decision rules. Can be either:</br>
        - `"mean"`: outputs the target mean within the partitions defined by the decision
        rules.</br>
        - `"model"`: always use instances of the model passed `pred_model` to make
        predictions.</br>
        - `"adaptive"`: dynamically selects between "mean" and "model" for each incoming example.
        The most accurate option at the moment will be used.
    pred_model
        The regression model that will be replicated for every rule when `pred_type` is either
        `"model"` or `"adaptive"`.
    splitter
        The Splitter or Attribute Observer (AO) used to monitor the class statistics of numeric
        features and perform splits. Splitters are available in the `tree.splitter` module.
        Different splitters are available for classification and regression tasks. Classification
        and regression splitters can be distinguished by their property `is_target_class`.
        This is an advanced option. Special care must be taken when choosing different splitters.
        By default, `tree.splitter.TEBSTSplitter` is used if `splitter` is `None`.
    drift_detector
        The drift detection model that is used by each rule. Care must be taken to avoid the
        triggering of too many false alarms or delaying too much the concept drift detection.
        By default, `drift.ADWIN` is used if `drift_detector` is `None`.
    fading_factor
        The exponential decaying factor applied to the learning models' absolute errors, that
        are monitored if `pred_type='adaptive'`. Must be between `0` and `1`. The closer
        to `1`, the more importance is going to be given to past observations. On the other hand,
        if its value approaches `0`, the recent observed errors are going to have more influence
        on the final decision.
    anomaly_threshold
        The threshold below which instances will be considered anomalies by the rules.
    m_min
        The minimum total weight a rule must observe before it starts to skip anomalous
        instances during training.
    ordered_rule_set
        If `True`, only the first rule that covers an instance will be used for training or
        prediction. If `False`, all the rules covering an instance will be updated during
        training, and the predictions for an instance will be the average prediction of all
        rules covering that example.
    min_samples_split
        The minimum number of samples each partition of a binary split candidate must have
        to be considered valid.

    Notes
    -----
    AMRules treats all the non-numerical inputs as nominal features. All instances of
    `numbers.Number` will be treated as continuous, even if they represent integer categories.
    When using nominal features, `pred_type` should be set to "mean", otherwise errors will be
    thrown while trying to update the underlying rules' prediction models. Prediction strategies
    other than "mean" can be used, as long as the prediction model passed to `pred_model` supports
    nominal features.

    Raises
    ------
    TypeError
        If one or more input features are non-numeric and the selected `pred_model` is either
        "model" or "adaptive".

    Examples
    --------
    >>> from river import datasets
    >>> from river import drift
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import preprocessing
    >>> from river import rules

    >>> dataset = datasets.TrumpApproval()

    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     rules.AMRules(
    ...         delta=0.01,
    ...         n_min=50,
    ...         drift_detector=drift.ADWIN()
    ...     )
    ... )

    >>> metric = metrics.MAE()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    MAE: 1.119553

    References
    ----------
    [^1]: Duarte, J., Gama, J. and Bifet, A., 2016. Adaptive model rules from high-speed data
    streams. ACM Transactions on Knowledge Discovery from Data (TKDD), 10(3), pp.1-22.

    """

    _PRED_MEAN = "mean"
    _PRED_MODEL = "model"
    _PRED_ADAPTIVE = "adaptive"
    _VALID_PRED = [_PRED_MEAN, _PRED_MODEL, _PRED_ADAPTIVE]

    def __init__(
        self,
        n_min: int = 200,
        delta: float = 1e-7,
        tau: float = 0.05,
        pred_type: str = "adaptive",
        pred_model: base.Regressor | None = None,
        splitter: spl.Splitter | None = None,
        drift_detector: base.DriftDetector | None = None,
        fading_factor: float = 0.99,
        anomaly_threshold: float = -0.75,
        m_min: int = 30,
        ordered_rule_set: bool = True,
        min_samples_split: int = 5,
    ):
        self.n_min = n_min
        self.delta = delta
        self.tau = tau

        if pred_type not in self._VALID_PRED:
            raise ValueError(f"Invalid 'pred_type': {pred_type}")
        self.pred_type = pred_type
        self.pred_model = pred_model if pred_model else linear_model.LinearRegression()

        if splitter is None:
            self.splitter = spl.TEBSTSplitter()
        else:
            self.splitter = splitter  # type: ignore

        self.drift_detector = drift_detector if drift_detector is not None else drift.ADWIN()

        self.fading_factor = fading_factor
        self.anomaly_threshold = anomaly_threshold
        self.m_min = m_min
        self.ordered_rule_set = ordered_rule_set
        self.min_samples_split = min_samples_split

        self._default_rule = self._new_rule()
        self._rules: dict[typing.Hashable, RegRule] = {}

        self._n_drifts_detected: int = 0

    @property
    def _mutable_attributes(self):
        return {
            "n_min",
            "delta",
            "tau",
            "fading_factor",
            "anomaly_threshold",
            "m_min",
            "ordered_rule_set",
        }

    def __len__(self):
        return len(self._rules) + 1

    def __getitem__(self, item):
        return list(self._rules.values())[item]

    @property
    def n_drifts_detected(self) -> int:
        """The number of detected concept drifts."""
        return self._n_drifts_detected

    def _new_rule(self) -> RegRule:
        if self.pred_type == self._PRED_MEAN:
            predictor = MeanRegressor()
        elif self.pred_type == self._PRED_MODEL:
            predictor = self.pred_model.clone()
        else:  # adaptive predictor
            predictor = AdaptiveRegressor(
                model_predictor=self.pred_model.clone(),
                fading_factor=self.fading_factor,
            )  # type: ignore

        return RegRule(
            template_splitter=self.splitter,
            split_criterion=split_criterion.VarianceRatioSplitCriterion(self.min_samples_split),
            pred_model=predictor,
            drift_detector=self.drift_detector.clone(),
        )

    def learn_one(self, x: dict, y: base.typing.RegTarget, w: int = 1) -> AMRules:
        any_covered = False
        to_del = set()

        for rule_id, rule in self._rules.items():
            if not rule.covers(x):
                continue

            # Anomaly detected skip training
            if rule.total_weight > self.m_min and rule.score_one(x) < self.anomaly_threshold:
                continue

            y_pred = rule.predict_one(x)

            in_drift = rule.drift_test(y, y_pred)  # noqa
            if in_drift:
                to_del.add(rule_id)
                self._n_drifts_detected += 1
                continue

            any_covered = True
            rule.learn_one(x, y, w)

            if rule.total_weight - rule.last_expansion_attempt_at >= self.n_min:
                updated_rule, expanded = rule.expand(self.delta, self.tau)

                if expanded:
                    updated_rule.pred_model = rule.pred_model
                    self._rules[rule_id] = updated_rule

            # Only the first matching rule is updated
            if self.ordered_rule_set:
                break

        if not any_covered:
            self._default_rule.learn_one(x, y, w)

            expanded = False
            if (
                self._default_rule.total_weight - self._default_rule.last_expansion_attempt_at
                >= self.n_min
            ):
                updated_rule, expanded = self._default_rule.expand(self.delta, self.tau)

            if expanded:
                updated_rule.pred_model = self._default_rule.pred_model  # noqa
                code = hash(updated_rule)
                self._rules[code] = updated_rule

                self._default_rule = self._new_rule()

        # Drop the rules affected by concept drift
        for rule_id in to_del:
            del self._rules[rule_id]

        return self

    def predict_one(self, x: dict) -> base.typing.RegTarget:
        y_pred = 0
        hits = 0

        for rule in self._rules.values():
            if rule.covers(x):
                y_pred += rule.predict_one(x)
                hits += 1

                if self.ordered_rule_set:
                    break

        if hits > 0:
            return y_pred / hits  # type: ignore
        else:
            return self._default_rule.predict_one(x)

    def anomaly_score(self, x) -> tuple[float, float, float]:
        """Aggregated anomaly score computed using all the rules that cover the input instance.

        Returns the mean anomaly score, the standard deviation of the score, and the proportion
        of rules that cover the instance (support). If the support is zero, it means that the
        default rule was used (not other rule covered `x`).

        Parameters
        ----------
        x
            The input instance.

        Returns
        -------
        mean_anomaly_score, std_anomaly_score, support

        Examples
        --------
        >>> from river import drift
        >>> from river import rules
        >>> from river import tree
        >>> from river.datasets import synth

        >>> dataset = synth.Friedman(seed=42).take(1001)

        >>> model = rules.AMRules(
        ...     n_min=50,
        ...     delta=0.1,
        ...     drift_detector=drift.ADWIN(),
        ...     splitter=tree.splitter.QOSplitter()
        ... )

        >>> for i, (x, y) in enumerate(dataset):
        ...     if i == 1000:
        ...         # Skip the last example
        ...         break
        ...     model = model.learn_one(x, y)

        >>> model.anomaly_score(x)
        (1.0168907243483933, 0.13045786430817402, 1.0)

        """
        var = stats.Var()

        for rule in self._rules.values():
            if rule.covers(x):
                var.update(rule.score_one(x))

        if var.mean.n > 0:
            return var.mean.get(), math.sqrt(var.get()), var.mean.n / len(self._rules)

        # No rule covers the instance. Use the default rule
        return self._default_rule.score_one(x), 0.0, 0

    def debug_one(self, x) -> str:
        """Return an explanation of how `x` is predicted

        Parameters
        ----------
        x
            The input instance.

        Returns
        -------
        A representation of the rules that cover the input and their prediction.

        Examples
        --------
        >>> from river import drift
        >>> from river import rules
        >>> from river import tree
        >>> from river.datasets import synth

        >>> dataset = synth.Friedman(seed=42).take(1001)

        >>> model = rules.AMRules(
        ...     n_min=50,
        ...     delta=0.1,
        ...     drift_detector=drift.ADWIN(),
        ...     splitter=tree.splitter.QOSplitter(),
        ...     ordered_rule_set=False
        ... )

        >>> for i, (x, y) in enumerate(dataset):
        ...     if i == 1000:
        ...         # Skip the last example
        ...         break
        ...     model = model.learn_one(x, y)

        >>> print(model.debug_one(x))
        Rule 0: 3 > 0.5060 and 0 > 0.2538
            Prediction (adaptive): 18.7217
        Rule 1: 1 > 0.2480 and 3 > 0.2573
            Prediction (adaptive): 19.4173
        Final prediction: 19.0695
        <BLANKLINE>

        """
        buffer = io.StringIO()
        _print = functools.partial(print, file=buffer)

        any_covered = False
        for i, rule in enumerate(self._rules.values()):
            if rule.covers(x):
                any_covered = True
                _print(f"Rule {i}: {repr(rule)}")
                _print(f"\tPrediction ({self.pred_type}): {rule.predict_one(x):.4f}")
                if self.ordered_rule_set:
                    break

        if any_covered:
            if not self.ordered_rule_set:
                _print(f"Final prediction: {self.predict_one(x):.4f}")
        else:
            _print("Default rule triggered:")
            _print(f"\tPrediction ({self.pred_type}): {self._default_rule.predict_one(x):.4f}")

        return buffer.getvalue()
