__author__ = 'Jacob Montiel'

from skmultiflow.classification.core.attribute_class_observers.attribute_class_observer import AttributeClassObserver
from skmultiflow.classification.core.attribute_split_suggestion import AttributeSplitSuggestion
from skmultiflow.classification.core.conditional_tests.nominal_attribute_binary_test import NominalAttributeBinaryTest
from skmultiflow.classification.core.conditional_tests.nominal_attribute_multiway_test import NominalAttributeMultiwayTest

class NominalAttributeClassObserver(AttributeClassObserver):
    """ NominalAttributeClassObserver

    Class for observing the class data distribution for a nominal attribute.
    This observer monitors the class distribution of a given attribute.
    Used in naive Bayes and decision trees to monitor data statistics on leaves.

    """
    def __init__(self):
        super().__init__()
        self._total_weight_observed = 0.0
        self._missing_weight_observed = 0.0
        self._att_val_dist_per_class = {}

    def observe_attribute_class(self, att_val, class_val, weight):
        if att_val is None:
            self._missing_weight_observed += weight
        else:
            if class_val not in self._att_val_dist_per_class:
                self._att_val_dist_per_class[class_val] = {}
            val_dist = self._att_val_dist_per_class[class_val]
            if int(att_val) not in val_dist:
                val_dist[int(att_val)] = 0.0
            val_dist[int(att_val)] += weight
        self._total_weight_observed += weight

    def probability_of_attribute_value_given_class(self, att_val, class_val):
        if class_val in self._att_val_dist_per_class:
            obs = self._att_val_dist_per_class[class_val]
            return obs[int(att_val)] + 1.0 / (sum(obs.values()) + len(obs))
        else:
            return 0.0

    def get_best_evaluated_split_suggestion(self, criterion, pre_split_dist, att_idx, binary_only):
        best_suggestion = None
        max_att_val_observed = self.get_max_att_val_observed()
        if not binary_only:
            post_split_dist = self.get_class_dist_from_multiway_split(max_att_val_observed)
            merit = criterion.get_merit_of_split(pre_split_dist, post_split_dist)
            nom_att_multiway_test = NominalAttributeMultiwayTest(att_idx)
            best_suggestion = AttributeSplitSuggestion(nom_att_multiway_test, post_split_dist, merit)
        for val_idx in range(max_att_val_observed):
            post_split_dist = self.get_class_dist_from_binary_split(val_idx)
            merit = criterion.get_merit_of_split(pre_split_dist, post_split_dist)
            if best_suggestion is None or merit > best_suggestion.merit:
                nom_att_binary_test = NominalAttributeBinaryTest(att_idx, val_idx)
                best_suggestion = AttributeSplitSuggestion(nom_att_binary_test, post_split_dist, merit)
        return best_suggestion

    def get_max_att_val_observed(self):
        max_att_val_observed = 0
        for _, att_val_dist in self._att_val_dist_per_class.items():
            if len(att_val_dist) > max_att_val_observed:
                max_att_val_observed = len(att_val_dist)
        return max_att_val_observed

    def get_class_dist_from_multiway_split(self, max_att_val_observed):
        resulting_dist = {}
        for i in range(max_att_val_observed):
            resulting_dist[i] = {}
        for i, att_val_dist in self._att_val_dist_per_class.items():
            for j, value in att_val_dist.items():
                if j not in resulting_dist[i]:
                    resulting_dist[i][j] = 0
                resulting_dist[i][j] += value
        return [resulting_dist]

    def get_class_dist_from_binary_split(self, val_idx):
        equal_dist = {}
        not_equal_dist = {}
        for i, att_val_dist in self._att_val_dist_per_class.items():
            for j, value in att_val_dist.items():
                if j == val_idx:
                    if i not in equal_dist:
                        equal_dist[i] = 0
                    equal_dist[i] += value
                else:
                    if i not in not_equal_dist:
                        not_equal_dist[i] = 0
                    not_equal_dist[i] += value
        return [equal_dist, not_equal_dist]