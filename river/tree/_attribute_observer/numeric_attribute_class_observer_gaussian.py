import typing
import math
from river.base.typing import ClfTarget

from river.proba import Gaussian
from .._attribute_test import NumericAttributeBinaryTest
from .._attribute_test import AttributeSplitSuggestion
from .attribute_observer import AttributeObserver


class NumericAttributeClassObserverGaussian(AttributeObserver):
    """Numeric attribute observer for classification tasks that is based on
    Gaussian estimators.


    Parameters
    ----------
    n_splits
        The number of partitions to consider when querying for split candidates.
    """

    def __init__(self, n_splits: int = 10):
        super().__init__()
        self._min_per_class: typing.Dict[ClfTarget, float] = {}
        self._max_per_class: typing.Dict[ClfTarget, float] = {}
        self._att_dist_per_class: typing.Dict[ClfTarget, Gaussian] = {}
        self.n_splits = n_splits

    def update(self, att_val, class_val, sample_weight):
        if att_val is None:
            return
        else:
            try:
                val_dist = self._att_dist_per_class[class_val]
                if att_val < self._min_per_class[class_val]:
                    self._min_per_class[class_val] = att_val
                if att_val > self._max_per_class[class_val]:
                    self._max_per_class[class_val] = att_val
            except KeyError:
                val_dist = Gaussian()
                self._att_dist_per_class[class_val] = val_dist
                self._min_per_class[class_val] = att_val
                self._max_per_class[class_val] = att_val

            val_dist.update(att_val, sample_weight)

        return self

    def probability_of_attribute_value_given_class(self, att_val, class_val):
        if class_val in self._att_dist_per_class:
            obs = self._att_dist_per_class[class_val]
            return obs.pdf(att_val)
        else:
            return 0.0

    def best_evaluated_split_suggestion(self, criterion, pre_split_dist, att_idx, binary_only):
        best_suggestion = None
        suggested_split_values = self._split_point_suggestions()
        for split_value in suggested_split_values:
            post_split_dist = self._class_dists_from_binary_split(split_value)
            merit = criterion.merit_of_split(pre_split_dist, post_split_dist)
            if best_suggestion is None or merit > best_suggestion.merit:
                num_att_binary_test = NumericAttributeBinaryTest(att_idx, split_value, True)
                best_suggestion = AttributeSplitSuggestion(
                    num_att_binary_test, post_split_dist, merit
                )
        return best_suggestion

    def _split_point_suggestions(self):
        suggested_split_values = []
        min_value = math.inf
        max_value = -math.inf
        for k, estimator in self._att_dist_per_class.items():
            if self._min_per_class[k] < min_value:
                min_value = self._min_per_class[k]
            if self._max_per_class[k] > max_value:
                max_value = self._max_per_class[k]
        if min_value < math.inf:
            bin_size = max_value - min_value
            bin_size /= self.n_splits + 1.0
            for i in range(self.n_splits):
                split_value = min_value + (bin_size * (i + 1))
                if min_value < split_value < max_value:
                    suggested_split_values.append(split_value)
        return suggested_split_values

    def _class_dists_from_binary_split(self, split_value):
        lhs_dist = {}
        rhs_dist = {}
        for k, estimator in self._att_dist_per_class.items():
            if split_value < self._min_per_class[k]:
                rhs_dist[k] = estimator.n_samples
            elif split_value >= self._max_per_class[k]:
                lhs_dist[k] = estimator.n_samples
            else:
                lhs_dist[k] = estimator.cdf(split_value) * estimator.n_samples
                rhs_dist[k] = estimator.n_samples - lhs_dist[k]
        return [lhs_dist, rhs_dist]
