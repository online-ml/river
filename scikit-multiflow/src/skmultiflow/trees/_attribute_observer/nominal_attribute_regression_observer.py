from skmultiflow.trees._attribute_test import NominalAttributeBinaryTest
from skmultiflow.trees._attribute_test import NominalAttributeMultiwayTest
from skmultiflow.trees._attribute_test import AttributeSplitSuggestion
from .attribute_observer import AttributeObserver


class NominalAttributeRegressionObserver(AttributeObserver):
    """ Class for observing the data distribution for a nominal attribute for
    regression.
    """

    def __init__(self):
        super().__init__()
        self._statistics = {}

    def update(self, att_val, target, weight=1.0):
        if att_val is None or weight is None:
            return
        else:
            try:
                self._statistics[att_val][0] += weight
                self._statistics[att_val][1] += weight * target
                self._statistics[att_val][2] += weight * target * target
            except KeyError:
                self._statistics[att_val] = {
                    0: weight,
                    1: weight * target,
                    2: weight * target * target
                }
                self._statistics = dict(
                    sorted(self._statistics.items())
                )

    def probability_of_attribute_value_given_class(self, att_val, target):
        return 0.0

    def get_best_evaluated_split_suggestion(self, criterion, pre_split_dist, att_idx, binary_only):
        current_best = None
        ordered_feature_values = sorted(list(self._statistics.keys()))
        if not binary_only:
            post_split_dist = [self._statistics[k] for k in ordered_feature_values]

            merit = criterion.get_merit_of_split(
                pre_split_dist, post_split_dist
            )
            branch_mapping = {attr_val: branch_id for branch_id, attr_val in
                              enumerate(ordered_feature_values)}
            current_best = AttributeSplitSuggestion(
                NominalAttributeMultiwayTest(att_idx, branch_mapping),
                post_split_dist, merit
            )

        for att_val in ordered_feature_values:
            actual_dist = self._statistics[att_val]
            remaining_dist = {
                0: pre_split_dist[0] - actual_dist[0],
                1: pre_split_dist[1] - actual_dist[1],
                2: pre_split_dist[2] - actual_dist[2]
            }
            post_split_dist = [actual_dist, remaining_dist]

            merit = criterion.get_merit_of_split(pre_split_dist,
                                                 post_split_dist)

            if current_best is None or merit > current_best.merit:
                nom_att_binary_test = NominalAttributeBinaryTest(att_idx,
                                                                 att_val)
                current_best = AttributeSplitSuggestion(
                    nom_att_binary_test, post_split_dist, merit
                )

        return current_best
