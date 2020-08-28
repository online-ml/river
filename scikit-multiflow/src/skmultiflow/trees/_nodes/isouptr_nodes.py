import numpy as np

from skmultiflow.utils import get_dimensions

from .base import InactiveLeaf
from .htr_nodes import ActiveLeafRegressor
from .htr_nodes import LearningNodePerceptron


class LearningNodePerceptronMultiTarget(LearningNodePerceptron):
    def __init__(self, initial_stats=None, parent_node=None, random_state=None):
        super().__init__(initial_stats, parent_node, random_state)

    def learn_one(self, X, y, *, weight=1.0, tree=None):
        """Update the node with the provided instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes for updating the node.
        y: numpy.ndarray of length equal to the number of targets.
            Instance targets.
        weight: float
            Instance weight.
        tree: HoeffdingTreeRegressor
            Regression Hoeffding Tree to update.
        """
        self.update_stats(y, weight)
        self.update_attribute_observers(X, y, weight, tree)

        if self.perceptron_weights is None:
            # Creates matrix of perceptron random weights
            _, rows = get_dimensions(y)
            _, cols = get_dimensions(X)

            self.perceptron_weights = self._random_state.uniform(-1.0, 1.0, (rows, cols + 1))
            self._normalize_perceptron_weights()

        if tree.learning_ratio_const:
            learning_ratio = tree.learning_ratio_perceptron
        else:
            learning_ratio = tree.learning_ratio_perceptron / (
                1 + self.stats[0] * tree.learning_ratio_decay)

        for i in range(int(weight)):
            self._update_weights(X, y, learning_ratio, tree)

    def predict_one(self, X, *, tree=None):
        if self.perceptron_weights is None or tree.examples_seen <= 1:
            return self.stats[1] / self.stats[0] if len(self.stats) > 0 else 0.0

        X_norm = tree.normalize_sample(X)
        Y_pred = np.matmul(self.perceptron_weights, X_norm)
        mean = tree.sum_of_values / tree.examples_seen
        variance = ((tree.sum_of_squares - (tree.sum_of_values * tree.sum_of_values)
                    / tree.examples_seen) / (tree.examples_seen - 1))
        sd = np.sqrt(variance, out=np.zeros_like(variance), where=variance >= 0.0)
        # Samples are normalized using just one sd, as proposed in the iSoup-Tree method
        return Y_pred * sd + mean

    def _update_weights(self, X, y, learning_ratio, tree):
        """ Update the perceptron weights

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes for updating the node.
        y: numpy.ndarray of length equal to the number of targets.
            Targets values.
        learning_ratio: float
            perceptron learning ratio
        tree: HoeffdingTreeRegressor
            Regression Hoeffding Tree to update.
        """
        X_norm = tree.normalize_sample(X)
        Y_pred = np.matmul(self.perceptron_weights, X_norm)

        Y_norm = tree.normalize_target_value(y)

        self.perceptron_weights += learning_ratio * np.matmul(
            (Y_norm - Y_pred)[:, None], X_norm[None, :])

        self._normalize_perceptron_weights()

    def _normalize_perceptron_weights(self):
        # Normalize perceptron weights
        n_targets = self.perceptron_weights.shape[0]
        for i in range(n_targets):
            sum_w = np.sum(np.abs(self.perceptron_weights[i, :]))
            self.perceptron_weights[i, :] /= sum_w


class LearningNodeAdaptiveMultiTarget(LearningNodePerceptronMultiTarget):
    def __init__(self, initial_stats=None, parent_node=None, random_state=None):
        super().__init__(initial_stats, parent_node, random_state)
        # Faded errors for the perceptron and mean predictors
        self.fMAE_M = 0.0
        self.fMAE_P = 0.0

    def predict_one(self, X, *, tree=None):
        # Mean predictor
        pred_M = self.stats[1] / self.stats[0] if len(self.stats) > 0 else 0.0
        if self.perceptron_weights is None or tree.examples_seen <= 1:
            return pred_M
        else:
            predictions = np.zeros(self.perceptron_weights.shape[0])
            # Perceptron
            X_norm = tree.normalize_sample(X)
            normalized_prediction = np.matmul(self.perceptron_weights, X_norm)
            mean = tree.sum_of_values / tree.examples_seen
            variance = ((tree.sum_of_squares - (tree.sum_of_values * tree.sum_of_values)
                        / tree.examples_seen) / (tree.examples_seen - 1))
            sd = np.sqrt(variance, out=np.zeros_like(variance), where=variance >= 0.0)

            pred_P = normalized_prediction * sd + mean

            for j in range(self.perceptron_weights.shape[0]):
                if self.fMAE_P[j] <= self.fMAE_M[j]:
                    predictions[j] = pred_P[j]
                else:
                    predictions[j] = pred_M[j]

            return predictions

    def _update_weights(self, X, y, learning_ratio, tree):
        """Update the perceptron weights

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes for updating the node.
        y: numpy.ndarray of length equal to the number of targets.
            Targets values.
        learning_ratio: float
            perceptron learning ratio
        tree: HoeffdingTreeRegressor
            Regression Hoeffding Tree to update.
        """
        X_norm = tree.normalize_sample(X)
        Y_pred = np.matmul(self.perceptron_weights, X_norm)

        Y_norm = tree.normalize_target_value(y)

        self.perceptron_weights += learning_ratio * np.matmul(
            (Y_norm - Y_pred)[:, None], X_norm[None, :])

        self._normalize_perceptron_weights()

        # Update faded errors for the predictors
        # The considered errors are normalized, since they are based on
        # mean centered and std scaled values
        self.fMAE_P = 0.95 * self.fMAE_P + np.abs(Y_norm - Y_pred)

        pred_mean = self.stats[1] / self.stats[0] if len(self.stats) > 0 else np.zeros_like(y)
        self.fMAE_M = 0.95 * self.fMAE_M + np.abs(
            Y_norm - tree.normalize_target_value(pred_mean))


class ActiveLearningNodePerceptronMultiTarget(LearningNodePerceptronMultiTarget,
                                              ActiveLeafRegressor):
    """ Learning Node for Multi-target Regression tasks that always use linear
    perceptron predictors for each target.

    Parameters
    ----------
    initial_stats: dict
        In regression tasks this dictionary carries the sufficient to perform
        online variance calculation. They refer to the number of observations
        (key '0'), the sum of the targets values (key '1'), and the sum of the
        squared targets values (key '2').
    parent_node: ActiveLearningNodePerceptronMultiTarget (default=None)
        A node containing statistics about observed data.
    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    def __init__(self, initial_stats=None, parent_node=None, random_state=None):
        """ActiveLearningNodePerceptronMultiTarget class constructor."""
        super().__init__(initial_stats, parent_node, random_state)


class InactiveLearningNodePerceptronMultiTarget(LearningNodePerceptronMultiTarget, InactiveLeaf):
    """ Inactive Learning Node for Multi-target Regression tasks that always use
    linear perceptron predictors for each target.

    Parameters
    ----------
    initial_stats: dict
        In regression tasks this dictionary carries the sufficient to perform
        online variance calculation. They refer to the number of observations
        (key '0'), the sum of the targets values (key '1'), and the sum of the
        squared targets values (key '2').
    parent_node: ActiveLearningNodePerceptronMultiTarget (default=None)
        A node containing statistics about observed data.
    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    def __init__(self, initial_stats=None, parent_node=None, random_state=None):
        """ InactiveLearningNodeForRegression class constructor."""
        super().__init__(initial_stats, parent_node, random_state)


class ActiveLearningNodeAdaptiveMultiTarget(LearningNodeAdaptiveMultiTarget, ActiveLeafRegressor):
    """ Learning Node for Multi-target Regression tasks that keeps track of both
    a linear perceptron and an average predictor for each target.

    Parameters
    ----------
    initial_stats: dict
        In regression tasks this dictionary carries the sufficient to perform
        online variance calculation. They refer to the number of observations
        (key '0'), the sum of the targets values (key '1'), and the sum of the
        squared targets values (key '2').
    parent_node: ActiveLearningNodeAdaptiveMultiTarget (default=None)
        A node containing statistics about observed data.
    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    def __init__(self, initial_stats=None, parent_node=None, random_state=None):
        """ActiveLearningNodeAdaptiveMultiTarget class constructor."""
        super().__init__(initial_stats, parent_node, random_state)


class InactiveLearningNodeAdaptiveMultiTarget(LearningNodeAdaptiveMultiTarget, InactiveLeaf):
    """ Inactive Learning Node for Multi-target Regression tasks that keeps
    track of both a linear perceptron and an average predictor for each target.

    Parameters
    ----------
    initial_stats: dict
        In regression tasks this dictionary carries the sufficient to perform
        online variance calculation. They refer to the number of observations
        (key '0'), the sum of the target values (key '1'), and the sum of the
        squared target values (key '2').
    parent_node: ActiveLearningNodeAdaptiveMultiTarget (default=None)
        A node containing statistics about observed data.
    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    def __init__(self, initial_stats=None, parent_node=None, random_state=None):
        """InactiveLearningNodeAdaptiveMultiTarget class constructor."""
        super().__init__(initial_stats, parent_node, random_state)
