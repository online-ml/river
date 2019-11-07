import numpy as np
from skmultiflow.trees.nodes import SSTInactiveLearningNode
from skmultiflow.utils import get_dimensions


class SSTInactiveLearningNodeAdaptive(SSTInactiveLearningNode):
    """ Inactive Multi-target regression learning node for SST-HT that keeps
    track of mean, perceptron, and stacked perceptrons predictors for each
    target.

    Parameters
    ----------
    initial_class_observations: dict
        A dictionary containing the set of sufficient statistics to be
        stored by the leaf node. It contains the following elements:
        - 0: the sum of elements seen so far;
        - 1: the sum of the targets values seen so far;
        - 2: the sum of the squared values of the targets seen so far.
    perceptron_weight: `numpy.ndarray` with number of features rows and
    number of targets columns.
        The weight matrix for the perceptron predictors. It will be
        extracted from the ActiveLearningNode which is being
        deactivated.
    random_state : `int`, `RandomState` instance or None (default=None)
        If int, `random_state` is used as seed to the random number
        generator; If a `RandomState` instance, `random_state` is the
        random number generator; If `None`, the random number generator
        is the current `RandomState` instance used by `np.random`.
    """
    def __init__(self, initial_class_observations, perceptron_weight=None,
                 random_state=None):
        """ SSTInactiveLearningNodeAdaptive class constructor."""

        super().__init__(initial_class_observations, perceptron_weight,
                         random_state)

        # Faded adaptive errors
        self.fMAE_M = 0.0
        self.fMAE_P = 0.0
        # Stacked Perceptron
        self.fMAE_SP = 0.0

    def update_weights(self, X, y, learning_ratio, rht):
        """Update the perceptron weights

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes for updating the node.
        y: numpy.ndarray of length equal to the number of targets.
            Targets values.
        learning_ratio: float
            perceptron learning ratio
        rht: RegressionHoeffdingTree
            Regression Hoeffding Tree to update.
        """
        normalized_sample = rht.normalize_sample(X)
        normalized_base_pred = self._predict_base(normalized_sample)

        _, n_features = get_dimensions(X)
        _, n_targets = get_dimensions(y)

        normalized_target_value = rht.normalized_target_value(y)

        self.perceptron_weight[0] += learning_ratio * \
            (normalized_target_value - normalized_base_pred)[:, None] @ \
            normalized_sample[None, :]

        # Add bias term
        normalized_base_pred = np.append(normalized_base_pred, 1.0)
        normalized_meta_pred = self._predict_meta(normalized_base_pred)

        self.perceptron_weight[1] += learning_ratio * \
            (normalized_target_value - normalized_meta_pred)[:, None] @ \
            normalized_base_pred[None, :]

        self.normalize_perceptron_weights()

        # Update faded errors for the predictors
        # The considered errors are normalized, since they are based on
        # mean centered and sd scaled values
        self.fMAE_M = 0.95 * self.fMAE_M + np.absolute(
            normalized_target_value - rht.
            normalized_target_value(self._observed_class_distribution[1] /
                                    self._observed_class_distribution[0])
        )

        # Ignore added bias term in the comparison
        self.fMAE_P = 0.95 * self.fMAE_P + np.absolute(
            normalized_target_value - normalized_base_pred[:-1]
        )

        self.fMAE_SP = 0.95 * self.fMAE_SP + np.absolute(
            normalized_target_value - normalized_meta_pred
        )
