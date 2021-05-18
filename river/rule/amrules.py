import collections
import copy
import functools
import io
import math
import numbers
import typing

from river import base, drift, linear_model, stats, tree

from ..tree.split_criterion import VarianceRatioSplitCriterion
from ..tree.splitter.nominal_splitter_reg import NominalSplitterReg
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
    def __init__(self, model_predictor: base.Regressor, alpha: float):
        self.model_predictor = model_predictor
        self.mean_predictor = MeanRegressor()
        self.alpha = alpha

        self._mae_mean = 0.0
        self._mae_model = 0.0

    def learn_one(self, x: dict, y: base.typing.RegTarget, w: int = 1):
        abs_error_mean = abs(y - self.mean_predictor.predict_one(x))  # noqa
        abs_error_model = abs(y - self.model_predictor.predict_one(x))  # noqa

        self._mae_mean = self.alpha * self._mae_mean + abs_error_mean
        self._mae_model = self.alpha * self._mae_model + abs_error_model

        self.mean_predictor.learn_one(x, y, w)

        try:
            self.model_predictor.learn_one(x, y, w)  # noqa
        except TypeError:
            for _ in range(int(w)):
                self.model_predictor.learn_one(x, y)

        return self

    def predict_one(self, x: dict):
        if self._mae_mean <= self._mae_model:
            return self.mean_predictor.predict_one(x)
        else:
            return self.model_predictor.predict_one(x)


class RegRule(HoeffdingRule, base.Regressor):
    def __init__(
        self, template_splitter, split_criterion, pred_model, drift_detector,
    ):
        super().__init__(
            template_splitter=template_splitter, split_criterion=split_criterion,
        )
        self.pred_model = pred_model
        self.drift_detector = drift_detector

        self._target_stats = stats.Var()
        self._feat_stats = collections.defaultdict(functools.partial(stats.Var))

    def new_nominal_splitter(self):
        return NominalSplitterReg()

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
        in_drift, _ = self.drift_detector.update(abs_error)

        return in_drift

    def anomaly_score(self, x):
        score = 0.0
        hits = 0

        for feat_name, feat_val in x.items():
            # Skip nominal features
            if not isinstance(feat_val, numbers.Number):
                continue

            mean = self._feat_stats[feat_name].mean.get()
            var = self._feat_stats[feat_name].get()
            std = math.sqrt(var)

            if std > 0:
                # One tailed variant of Chebyshev 's inequality
                k = abs(feat_val - mean) / std  # noqa
                proba = (2 * var) / (var + k * k)

                if 0 < proba < 1:
                    score += math.log(proba) - math.log(1 - proba)
                    hits += 1

        return score / hits if hits > 0 else 0.0

    def learn_one(self, x: dict, y: base.typing.RegTarget, w: int = 1):
        self.update(x, y, w)
        self.pred_model.learn_one(x, y, w)

        return self

    def predict_one(self, x: dict):
        return self.pred_model.predict_one(x)


class AMRules(base.Regressor):
    _PRED_MEAN = "mean"
    _PRED_MODEL = "model"
    _PRED_ADAPTIVE = "adaptive"
    _VALID_PRED = [_PRED_MEAN, _PRED_MODEL, _PRED_ADAPTIVE]

    def __init__(
        self,
        n_min: int = 200,
        tau: float = 0.05,
        delta: float = 1e-7,
        pred_type: str = "adaptive",
        pred_model: base.Regressor = None,
        splitter=None,
        drift_detector=None,
        model_selector_decay: float = 0.99,
        anomaly_threshold: float = -0.75,
        m_min: int = 30,
        ordered_rule_set=False,
        min_samples_split: int = 5,
    ):
        self.n_min = n_min
        self.tau = tau
        self.delta = delta

        if pred_type not in self._VALID_PRED:
            raise ValueError(f"Invalid 'pred_type': {pred_type}")
        self.pred_type = pred_type
        self.pred_model = pred_model if pred_model else linear_model.LinearRegression()

        if splitter is None:
            self.splitter = tree.splitter.EBSTSplitter()
        else:
            self.splitter = splitter

        self.drift_detector = (
            drift_detector
            if drift_detector is not None
            else drift.PageHinkley(threshold=35)
        )

        self.model_selector_decay = model_selector_decay
        self.anomaly_threshold = anomaly_threshold
        self.m_min = m_min
        self.ordered_rule_set = ordered_rule_set
        self.min_samples_split = min_samples_split

        self._default_rule = self._new_rule()
        self._rules: typing.Dict[typing.Hashable, RegRule] = {}

    def __len__(self):
        return len(self._rules) + 1

    def __getitem__(self, item):
        return list(self._rules.values())[item]

    def _new_rule(self) -> RegRule:
        if self.pred_type == self._PRED_MEAN:
            predictor = MeanRegressor()
        elif self.pred_type == self._PRED_MODEL:
            predictor = copy.deepcopy(self.pred_model)
        else:  # adaptive predictor
            predictor = AdaptiveRegressor(
                model_predictor=copy.deepcopy(self.pred_model),
                alpha=self.model_selector_decay,
            )

        return RegRule(
            template_splitter=self.splitter,
            split_criterion=VarianceRatioSplitCriterion(self.min_samples_split),
            pred_model=predictor,
            drift_detector=copy.deepcopy(self.drift_detector),
        )

    def learn_one(self, x: dict, y: base.typing.RegTarget, w: int = 1):
        any_covered = False
        to_del = set()

        for rule_id, rule in self._rules.items():
            if not rule.covers(x):
                continue

            if rule.total_weight > self.m_min:
                a_score = rule.anomaly_score(x)

                # Anomaly detected skip training
                if a_score < self.anomaly_threshold:
                    continue

            y_pred = rule.predict_one(x)

            in_drift = rule.drift_test(y, y_pred)  # noqa
            if in_drift:
                to_del.add(rule_id)
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
                self._default_rule.total_weight
                - self._default_rule.last_expansion_attempt_at
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
            return y_pred / hits
        else:
            return self._default_rule.predict_one(x)

    def anomaly_score(self, x):
        """

        Parameters
        ----------
        x
            The input instance.

        Returns
        -------
        mean_anomaly_score, std_anomaly_score, support
            The mean anomaly score, the standard deviation, and the proportion of rules that cover
            the instance.

        """
        var = stats.Var()

        for rule in self._rules.values():
            if rule.covers(x):
                var.update(rule.anomaly_score(x))

        if var.mean.n > 0:
            return var.mean.get(), math.sqrt(var.get()), var.mean.n / len(self._rules)

        # No rule covers the instance. Use the default rule
        return self._default_rule.anomaly_score(x), 0.0, 0

    def debug_one(self, x):
        buffer = io.StringIO()
        _print = functools.partial(print, file=buffer)

        any_covered = False
        for i, rule in self._rules.values():
            if rule.covers(x):
                any_covered = True
                _print(f"Rule {i}: {repr(rule)}")
                _print(f"\tPrediction: {rule.predict_one(x)}")
            if self.ordered_rule_set:
                break

        if any_covered:
            if not self.ordered_rule_set:
                _print(f"Final prediction: {self.predict_one(x)}")
        else:
            _print("Default rule triggered:")
            _print(f"\tPrediction: {self._default_rule.predict_one(x)}")

        return buffer.getvalue()
