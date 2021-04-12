import collections
import functools

from .._nodes.branch import BranchFactory
from .base_splitter import Splitter


class NominalSplitterClassif(Splitter):
    """Splitter utilized to monitor nominal features in classification tasks.

    As the monitored feature is nominal, it already has well-defined partitions. Hence,
    this splitter uses dictionary structures to keep class counts for each incoming category.
    """

    def __init__(self):
        super().__init__()
        self._total_weight_observed = 0.0
        self._missing_weight_observed = 0.0
        self._att_val_dist_per_class = collections.defaultdict(
            functools.partial(collections.defaultdict, float)
        )
        self._att_values = set()

    @property
    def is_numeric(self):
        return False

    def update(self, att_val, target_val, sample_weight):
        if att_val is None:
            self._missing_weight_observed += sample_weight
        else:
            self._att_values.add(att_val)
            self._att_val_dist_per_class[target_val][att_val] += sample_weight

        self._total_weight_observed += sample_weight

    def cond_proba(self, att_val, target_val):
        obs = self._att_val_dist_per_class[target_val]
        value = self._att_val_dist_per_class[att_val]
        try:
            return (value + 1.0) / (sum(obs.values()) + len(obs))
        except ZeroDivisionError:
            return 0.0

    def best_evaluated_split_suggestion(
        self, criterion, pre_split_dist, att_idx, binary_only
    ):
        best_suggestion = None

        if not binary_only:
            post_split_dist = self._class_dist_from_multiway_split()
            merit = criterion.merit_of_split(pre_split_dist, post_split_dist)

            best_suggestion = BranchFactory(
                merit,
                att_idx,
                sorted(self._att_values),
                post_split_dist,
                numerical_feature=False,
                multiway_split=True,
            )

        for att_val in self._att_values:
            post_split_dist = self._class_dist_from_binary_split(att_val)
            merit = criterion.merit_of_split(pre_split_dist, post_split_dist)
            if best_suggestion is None or merit > best_suggestion.merit:
                best_suggestion = BranchFactory(
                    merit,
                    att_idx,
                    att_val,
                    post_split_dist,
                    numerical_feature=False,
                    multiway_split=False,
                )

        return best_suggestion

    def _class_dist_from_multiway_split(self):
        resulting_dist = {}
        for i, att_val_dist in self._att_val_dist_per_class.items():
            for j, value in att_val_dist.items():
                if j not in resulting_dist:
                    resulting_dist[j] = {}
                if i not in resulting_dist[j]:
                    resulting_dist[j][i] = 0.0
                resulting_dist[j][i] += value

        sorted_keys = sorted(resulting_dist.keys())
        distributions = [dict(sorted(resulting_dist[k].items())) for k in sorted_keys]
        return distributions

    def _class_dist_from_binary_split(self, val_idx):
        equal_dist = {}
        not_equal_dist = {}
        for i, att_val_dist in self._att_val_dist_per_class.items():
            for j, value in att_val_dist.items():
                if j == val_idx:
                    if i not in equal_dist:
                        equal_dist[i] = 0.0
                    equal_dist[i] += value
                else:
                    if i not in not_equal_dist:
                        not_equal_dist[i] = 0.0
                    not_equal_dist[i] += value
        return [equal_dist, not_equal_dist]
