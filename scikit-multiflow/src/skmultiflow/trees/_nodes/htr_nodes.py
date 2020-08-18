from copy import deepcopy

import math
import numpy as np


from skmultiflow.utils import check_random_state
from skmultiflow.trees._attribute_observer import NominalAttributeRegressionObserver
from skmultiflow.trees._attribute_observer import NumericAttributeRegressionObserver
from .base import ActiveLeaf
from .base import InactiveLeaf
from .base import LearningNode


class ActiveLeafRegressor(ActiveLeaf):
    @staticmethod
    def new_nominal_attribute_observer():
        return NominalAttributeRegressionObserver()

    @staticmethod
    def new_numeric_attribute_observer():
        return NumericAttributeRegressionObserver()

    def manage_memory(self, criterion, last_check_ratio, last_check_sdr, last_check_e):
        """ Trigger Attribute Observers' memory management routines.

        Currently, only `NumericAttributeRegressionObserver` has support to this feature.

        Parameters
        ----------
            criterion: SplitCriterion
                Split criterion
            last_check_ratio: float
                The ratio between the second best candidate's merit and the merit of the best
                split candidate.
            last_check_sdr: float
                The best candidate's split merit.
            last_check_e: float
                Hoeffding bound value calculated in the last split attempt.
        """
        for obs in self.attribute_observers.values():
            if isinstance(obs, NumericAttributeRegressionObserver):
                obs.remove_bad_splits(criterion=criterion, last_check_ratio=last_check_ratio,
                                      last_check_sdr=last_check_sdr, last_check_e=last_check_e,
                                      pre_split_dist=self._stats)


class LearningNodeMean(LearningNode):
    def __init__(self, initial_stats=None):
        super().__init__(initial_stats)

    def update_stats(self, y, weight):
        try:
            self._stats[0] += weight
            self._stats[1] += y * weight
            self._stats[2] += y * y * weight
        except KeyError:
            self._stats[0] = weight
            self._stats[1] = y * weight
            self._stats[2] = y * y * weight

    def predict_one(self, X, *, tree=None):
        return self._stats[1] / self._stats[0] if len(self._stats) > 0 else 0.0

    @property
    def total_weight(self):
        """ Calculate the total weight seen by the node.

        Returns
        -------
        float
            Total weight seen.

        """
        return self._stats[0] if len(self._stats) > 0 else 0.0


class LearningNodePerceptron(LearningNodeMean):
    def __init__(self, initial_stats=None, parent_node=None, random_state=None):
        super().__init__(initial_stats)
        self.random_state = random_state
        self._random_state = check_random_state(self.random_state)
        if parent_node is None:
            self.perceptron_weights = None
        else:
            self.perceptron_weights = deepcopy(parent_node.perceptron_weights)

    def learn_one(self, X, y, *, weight=1.0, tree=None):
        super().learn_one(X, y, weight=weight, tree=tree)

        if self.perceptron_weights is None:
            self.perceptron_weights = self._random_state.uniform(-1, 1, len(X) + 1)

        if tree.learning_ratio_const:
            learning_ratio = tree.learning_ratio_perceptron
        else:
            learning_ratio = (tree.learning_ratio_perceptron
                              / (1 + self.stats[0] * tree.learning_ratio_decay))

        # Loop for compatibility with bagging methods
        for i in range(int(weight)):
            self._update_weights(X, y, learning_ratio, tree)

    def predict_one(self, X, *, tree=None):
        if self.perceptron_weights is None:
            return super().predict_one(X)

        X_norm = tree.normalize_sample(X)
        y_norm = self.perceptron_weights.dot(X_norm)
        # De-normalize prediction
        mean = tree.sum_of_values / tree.samples_seen
        sd = compute_sd(tree.sum_of_squares, tree.sum_of_values, tree.samples_seen)
        return y_norm * sd * 3 + mean

    def _update_weights(self, X, y, learning_ratio, tree):
        """
        Update the perceptron weights
        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes for updating the node.
        y: float
            Instance target value.
        learning_ratio: float
            perceptron learning ratio
        tree: HoeffdingTreeRegressor
            Regression Hoeffding Tree to update.
        """
        normalized_sample = tree.normalize_sample(X)
        normalized_pred = np.dot(self.perceptron_weights, normalized_sample)
        normalized_target_value = tree.normalize_target_value(y)
        delta = normalized_target_value - normalized_pred
        self.perceptron_weights = (self.perceptron_weights
                                   + learning_ratio * delta * normalized_sample)
        # Normalize perceptron weights
        self.perceptron_weights = self.perceptron_weights / np.sum(np.abs(self.perceptron_weights))


class ActiveLearningNodeMean(LearningNodeMean, ActiveLeafRegressor):
    """ Learning Node for regression tasks that always use the average target
    value as response.

    Parameters
    ----------
    initial_stats: dict
        In regression tasks this dictionary carries the sufficient to perform
        online variance calculation. They refer to the number of observations
        (key '0'), the sum of the target values (key '1'), and the sum of the
        squared target values (key '2').
    """
    def __init__(self, initial_stats=None):
        super().__init__(initial_stats)


class InactiveLearningNodeMean(LearningNodeMean, InactiveLeaf):
    """ Inactive Learning Node for regression tasks that always use
    the average target value as response.

    Parameters
    ----------
    initial_stats: dict
        In regression tasks this dictionary carries the sufficient to perform
        online variance calculation. They refer to the number of observations
        (key '0'), the sum of the target values (key '1'), and the sum of the
        squared target values (key '2').
    """
    def __init__(self, initial_stats=None):
        super().__init__(initial_stats)


class ActiveLearningNodePerceptron(LearningNodePerceptron, ActiveLeafRegressor):
    """ Learning Node for regression tasks that always use a linear perceptron
    model to provide responses.

    Parameters
    ----------
    initial_stats: dict
        In regression tasks this dictionary carries the sufficient statistics
        to perform online variance calculation. They refer to the number of
        observations (key '0'), the sum of the target values (key '1'), and
        the sum of the squared target values (key '2').
    parent_node: ActiveLearningNodePerceptron (default=None)
        A node containing statistics about observed data.
    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    def __init__(self, initial_stats=None, parent_node=None, random_state=None):
        super().__init__(initial_stats, parent_node, random_state)


class InactiveLearningNodePerceptron(LearningNodePerceptron, InactiveLeaf):
    """ Inactive Learning Node for regression tasks that always use a linear
    perceptron model to provide responses.

    Parameters
    ----------
    initial_stats: dict
        In regression tasks this dictionary carries the sufficient statistics
        to perform online variance calculation. They refer to the number of
        observations (key '0'), the sum of the target values (key '1'), and
        the sum of the squared target values (key '2').
    parent_node: ActiveLearningNodePerceptron (default=None)
        A node containing statistics about observed data.
    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    def __init__(self, initial_stats=None, parent_node=None, random_state=None):
        super().__init__(initial_stats, parent_node, random_state)


def compute_sd(square_val: float, val: float, size: float):
    if size > 1:
        a = square_val - ((val * val) / size)
        if a > 0:
            return math.sqrt(a / size)
    return 0.0
