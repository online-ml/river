from __future__ import annotations

import functools

from river.stats import Var
from river.utils import VectorDict

from ..utils import BranchFactory
from .base import Splitter


class NominalSplitterReg(Splitter):
    """Splitter utilized to monitor nominal features in regression tasks.

    As the monitored feature is nominal, it already has well-defined partitions. Hence,
    this splitter simply keeps the split-enabling target statistics for each incoming
    category."""

    def __init__(self):
        super().__init__()
        self._statistics = {}
        # The way in which the target statistics are update are selected on the fly.
        # Different strategies are used for univariate and multivariate regression.
        self._update_estimator = self._update_estimator_univariate

    @property
    def is_numeric(self):
        return False

    @staticmethod
    def _update_estimator_univariate(estimator, target, sample_weight):
        estimator.update(target, sample_weight)

    @staticmethod
    def _update_estimator_multivariate(estimator, target, sample_weight):
        for t in target:
            estimator[t].update(target[t], sample_weight)

    def update(self, att_val, target_val, sample_weight):
        if att_val is None or sample_weight is None:
            return
        else:
            try:
                estimator = self._statistics[att_val]
            except KeyError:
                if isinstance(target_val, dict):  # Multi-target case
                    self._statistics[att_val] = VectorDict(default_factory=functools.partial(Var))
                    self._update_estimator = self._update_estimator_multivariate
                else:
                    self._statistics[att_val] = Var()
                estimator = self._statistics[att_val]
            self._update_estimator(estimator, target_val, sample_weight)

    def cond_proba(self, att_val, target_val):
        """Not implemented in regression splitters."""
        raise NotImplementedError

    def best_evaluated_split_suggestion(self, criterion, pre_split_dist, att_idx, binary_only):
        current_best = BranchFactory()
        ordered_feature_values = sorted(list(self._statistics.keys()))
        if not binary_only and len(self._statistics) > 2:
            post_split_dist = [self._statistics[k] for k in ordered_feature_values]

            merit = criterion.merit_of_split(pre_split_dist, post_split_dist)
            current_best = BranchFactory(
                merit,
                att_idx,
                ordered_feature_values,
                post_split_dist,
                numerical_feature=False,
                multiway_split=True,
            )

        for att_val in ordered_feature_values:
            actual_dist = self._statistics[att_val]
            remaining_dist = pre_split_dist - actual_dist
            post_split_dist = [actual_dist, remaining_dist]

            merit = criterion.merit_of_split(pre_split_dist, post_split_dist)

            if merit > current_best.merit:
                current_best = BranchFactory(
                    merit,
                    att_idx,
                    att_val,
                    post_split_dist,
                    numerical_feature=False,
                    multiway_split=False,
                )

        return current_best
