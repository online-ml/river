from river.stats import Var
from river.utils import VectorDict

from .._attribute_test import NominalBinaryTest, NominalMultiwayTest, SplitSuggestion
from .base_splitter import Splitter


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
                    self._statistics[att_val] = VectorDict(
                        default_factory=lambda: Var()
                    )
                    self._update_estimator = self._update_estimator_multivariate
                else:
                    self._statistics[att_val] = Var()
                estimator = self._statistics[att_val]
            self._update_estimator(estimator, target_val, sample_weight)

        return self

    def cond_proba(self, att_val, target_val):
        """Not implemented in regression splitters."""
        raise NotImplementedError

    def best_evaluated_split_suggestion(
        self, criterion, pre_split_dist, att_idx, binary_only
    ):
        current_best = None
        ordered_feature_values = sorted(list(self._statistics.keys()))
        if not binary_only:
            post_split_dist = [self._statistics[k] for k in ordered_feature_values]

            merit = criterion.merit_of_split(pre_split_dist, post_split_dist)
            branch_mapping = {
                attr_val: branch_id
                for branch_id, attr_val in enumerate(ordered_feature_values)
            }
            current_best = SplitSuggestion(
                NominalMultiwayTest(att_idx, branch_mapping), post_split_dist, merit,
            )

        for att_val in ordered_feature_values:
            actual_dist = self._statistics[att_val]
            remaining_dist = pre_split_dist - actual_dist
            post_split_dist = [actual_dist, remaining_dist]

            merit = criterion.merit_of_split(pre_split_dist, post_split_dist)

            if current_best is None or merit > current_best.merit:
                nom_att_binary_test = NominalBinaryTest(att_idx, att_val)
                current_best = SplitSuggestion(
                    nom_att_binary_test, post_split_dist, merit
                )

        return current_best
