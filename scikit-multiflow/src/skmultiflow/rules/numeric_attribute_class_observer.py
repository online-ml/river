import numpy as np

from skmultiflow.rules.attribute_expand_suggestion import AttributeExpandSuggestion
from skmultiflow.trees._attribute_observer import AttributeObserver
from skmultiflow.trees.gaussian_estimator import GaussianEstimator


class GaussianNumericAttributeClassObserver(AttributeObserver):
    """ GaussianNumericAttributeClassObserver

    Class for observing the class data distribution for a numeric attribute using gaussian estimators.
    This observer monitors the class distribution of a given attribute.

    """

    def __init__(self):
        super().__init__()
        self._min_value_observed_per_class = {}
        self._max_value_observed_per_class = {}
        self._att_val_dist_per_class = {}
        self.num_bin_options = 10  # The number of bins, default 10

    def update(self, att_val, class_val, weight):
        if att_val is None:
            return
        else:
            try:
                val_dist = self._att_val_dist_per_class[class_val]
                if att_val < self._min_value_observed_per_class[class_val]:
                    self._min_value_observed_per_class[class_val] = att_val
                if att_val > self._max_value_observed_per_class[class_val]:
                    self._max_value_observed_per_class[class_val] = att_val
            except KeyError:
                val_dist = GaussianEstimator()
                self._att_val_dist_per_class[class_val] = val_dist
                self._min_value_observed_per_class[class_val] = att_val
                self._max_value_observed_per_class[class_val] = att_val
                self._att_val_dist_per_class = dict(sorted(self._att_val_dist_per_class.items()))
                self._max_value_observed_per_class = dict(sorted(self._max_value_observed_per_class.items()))
                self._min_value_observed_per_class = dict(sorted(self._min_value_observed_per_class.items()))
            val_dist.add_observation(att_val, weight)

    def probability_of_attribute_value_given_class(self, att_val, class_val):
        if class_val in self._att_val_dist_per_class:
            obs = self._att_val_dist_per_class[class_val]
            return obs.probability_density(att_val)
        else:
            return 0.0

    def get_best_evaluated_split_suggestion(self, criterion, pre_split_dist, att_idx, class_idx=None):
        best_suggestion = None
        suggested_split_values = self.get_split_point_suggestions()
        for expand_value in suggested_split_values:
            post_split_dist = self.get_class_dists_from_binary_split(expand_value)
            if class_idx is not None:
                criterion.class_idx = class_idx
            merit = criterion.get_merit_of_split(pre_split_dist, post_split_dist)
            if best_suggestion is None or merit > best_suggestion.merit:
                if criterion.best_idx == 0:
                    symbol = "<="
                else:
                    symbol = ">"
                best_suggestion = AttributeExpandSuggestion(att_idx, expand_value, symbol, post_split_dist, merit)
        return best_suggestion

    def get_split_point_suggestions(self):
        suggested_split_values = []
        min_value = np.inf
        max_value = -np.inf
        for k, estimator in self._att_val_dist_per_class.items():
            if self._min_value_observed_per_class[k] < min_value:
                min_value = self._min_value_observed_per_class[k]
            if self._max_value_observed_per_class[k] > max_value:
                max_value = self._max_value_observed_per_class[k]
        if min_value < np.inf:
            bin_size = max_value - min_value
            bin_size /= (float(self.num_bin_options) + 1.0)
            for i in range(self.num_bin_options):
                split_value = min_value + (bin_size * (i + 1))
                if split_value > min_value and split_value < max_value:
                    suggested_split_values.append(split_value)
        suggested_split_values.sort()
        return suggested_split_values

    def get_class_dists_from_binary_split(self, split_value):
        """
        Assumes all values equal to split_value go to lhs

        """

        lhs_dist = {}
        rhs_dist = {}
        for k, estimator in self._att_val_dist_per_class.items():
            if split_value < self._min_value_observed_per_class[k]:
                rhs_dist[k] = estimator.get_total_weight_observed()
            elif split_value >= self._max_value_observed_per_class[k]:
                lhs_dist[k] = estimator.get_total_weight_observed()
            else:
                weight_dist = estimator.estimated_weight_lessthan_equalto_greaterthan_value(split_value)
                lhs_dist[k] = weight_dist[0] + weight_dist[1]
                rhs_dist[k] = weight_dist[2]
        return [lhs_dist, rhs_dist]
