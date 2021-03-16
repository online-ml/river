from .._attribute_test import NominalBinaryTest, NominalMultiwayTest, SplitSuggestion
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
        self._att_val_dist_per_class = {}

    @property
    def is_numeric(self):
        return False

    def update(self, att_val, target_val, sample_weight):
        if att_val is None:
            self._missing_weight_observed += sample_weight
        else:
            try:
                self._att_val_dist_per_class[target_val]
            except KeyError:
                self._att_val_dist_per_class[target_val] = {att_val: 0.0}
                self._att_val_dist_per_class = dict(
                    sorted(self._att_val_dist_per_class.items())
                )
            try:
                self._att_val_dist_per_class[target_val][att_val] += sample_weight
            except KeyError:
                self._att_val_dist_per_class[target_val][att_val] = sample_weight
                self._att_val_dist_per_class[target_val] = dict(
                    sorted(self._att_val_dist_per_class[target_val].items())
                )

        self._total_weight_observed += sample_weight

        return self

    def cond_proba(self, att_val, target_val):
        obs = self._att_val_dist_per_class.get(target_val, None)
        if obs is not None:
            value = obs[att_val] if att_val in obs else 0.0
            return (value + 1.0) / (sum(obs.values()) + len(obs))
        return 0.0

    def best_evaluated_split_suggestion(
        self, criterion, pre_split_dist, att_idx, binary_only
    ):
        best_suggestion = None
        att_values = sorted(
            set(
                [
                    att_val
                    for att_val_per_class in self._att_val_dist_per_class.values()
                    for att_val in att_val_per_class
                ]
            )
        )
        if not binary_only:
            post_split_dist = self._class_dist_from_multiway_split()
            merit = criterion.merit_of_split(pre_split_dist, post_split_dist)
            branch_mapping = {
                attr_val: branch_id for branch_id, attr_val in enumerate(att_values)
            }
            best_suggestion = SplitSuggestion(
                NominalMultiwayTest(att_idx, branch_mapping), post_split_dist, merit,
            )
        for att_val in att_values:
            post_split_dist = self._class_dist_from_binary_split(att_val)
            merit = criterion.merit_of_split(pre_split_dist, post_split_dist)
            if best_suggestion is None or merit > best_suggestion.merit:
                best_suggestion = SplitSuggestion(
                    NominalBinaryTest(att_idx, att_val), post_split_dist, merit
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
