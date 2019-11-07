import numpy as np
from skmultiflow.trees.nodes import InactiveLearningNodePerceptronMultiTarget


class InactiveLearningNodeAdaptiveMultiTarget(InactiveLearningNodePerceptronMultiTarget):
    """ Inactive Learning Node for Multi-target Regression tasks that keeps
    track of both a linear perceptron and an average predictor for each target.

    Parameters
    ----------
    initial_class_observations: dict
        In regression tasks this dictionary carries the sufficient to perform
        online variance calculation. They refer to the number of observations
        (key '0'), the sum of the target values (key '1'), and the sum of the
        squared target values (key '2').
    perceptron_weight: np.ndarray(n_targets, n_features) or None, optinal
        (default=None)
        The weights for the linear models that predict the targets values. If
        not passed, uniform values in the range [-1, 1] are used.
    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    def __init__(self, initial_class_observations, perceptron_weight=None,
                 random_state=None):
        """InactiveLearningNodeAdaptiveMultiTarget class constructor."""
        super().__init__(initial_class_observations, perceptron_weight,
                         random_state)

        # Faded errors for the perceptron and mean predictors
        self.fMAE_M = 0.0
        self.fMAE_P = 0.0

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
        normalized_pred = self.predict(normalized_sample)

        normalized_target_value = rht.normalized_target_value(y)
        self.perceptron_weight += learning_ratio * \
            np.matmul((normalized_target_value - normalized_pred)[:, None],
                      normalized_sample[None, :])

        self.normalize_perceptron_weights()

        # Update faded errors for the predictors
        # The considered errors are normalized, since they are based on
        # mean centered and sd scaled values
        self.fMAE_P = 0.95 * self.fMAE_P + np.abs(
            normalized_target_value - normalized_pred
        )

        self.fMAE_M = 0.95 * self.fMAE_M + np.abs(
            normalized_target_value - rht.
            normalized_target_value(self._observed_class_distribution[1] /
                                    self._observed_class_distribution[0])
        )
