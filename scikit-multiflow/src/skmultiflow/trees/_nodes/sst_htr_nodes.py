import numpy as np

from skmultiflow.utils import get_dimensions
from .base import InactiveLeaf
from .htr_nodes import ActiveLeafRegressor
from .isouptr_nodes import LearningNodePerceptronMultiTarget


class SSTLearningNode(LearningNodePerceptronMultiTarget):
    def __init__(self, initial_stats=None, parent_node=None, random_state=None):
        super().__init__(initial_stats, parent_node, random_state)

    def learn_one(self, X, y, *, weight=None, tree=None):
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
            self.perceptron_weights = {}
            # Creates matrix of perceptron random weights
            _, rows = get_dimensions(y)
            _, cols = get_dimensions(X)

            self.perceptron_weights[0] = self._random_state.uniform(-1.0, 1.0, (rows, cols + 1))
            # Cascade Stacking
            self.perceptron_weights[1] = self._random_state.uniform(-1.0, 1.0, (rows, rows + 1))
            self._normalize_perceptron_weights()

        if tree.learning_ratio_const:
            learning_ratio = tree.learning_ratio_perceptron
        else:
            learning_ratio = (tree.learning_ratio_perceptron / (1 + self._stats[0]
                              * tree.learning_ratio_decay))

        for i in range(int(weight)):
            self._update_weights(X, y, learning_ratio, tree)

    def predict_one(self, X, *, tree=None):
        if self.perceptron_weights is None:
            return self.stats[1] / self.stats[0] if len(self.stats) > 0 else np.zeros(
                tree._n_targets, dtype=np.float64)

        X_norm = tree.normalize_sample(X)
        Y_pred_base = self._predict_one_base(X_norm)
        Y_pred_meta = self._predict_one_meta(np.append(Y_pred_base, 1.0))
        mean = tree.sum_of_values / tree.examples_seen
        variance = ((tree.sum_of_squares - (tree.sum_of_values * tree.sum_of_values)
                    / tree.examples_seen) / (tree.examples_seen - 1))
        sd = np.sqrt(variance, out=np.zeros_like(variance), where=variance >= 0.0)
        # Samples are normalized using just one std, as proposed in the iSoup-Tree method
        return Y_pred_meta * sd + mean

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
        Y_pred_base = self._predict_one_base(X_norm)

        Y_norm = tree.normalize_target_value(y)

        self.perceptron_weights[0] += learning_ratio * (Y_norm - Y_pred_base)[:, None] \
            @ X_norm[None, :]

        # Add bias term
        Y_pred_base = np.append(Y_pred_base, 1.0)
        Y_pred_meta = self._predict_one_meta(Y_pred_base)

        self.perceptron_weights[1] += learning_ratio * (Y_norm - Y_pred_meta)[:, None] \
            @ Y_pred_base[None, :]

        self._normalize_perceptron_weights()

    # Normalize both levels
    def _normalize_perceptron_weights(self):
        """ Normalizes both levels of perceptron weights."""
        n_targets = self.perceptron_weights[0].shape[0]
        # Normalize perceptron weights
        for i in range(n_targets):
            sum_w_0 = np.sum(np.absolute(self.perceptron_weights[0][i, :]))
            self.perceptron_weights[0][i, :] /= sum_w_0
            sum_w_1 = np.sum(np.absolute(self.perceptron_weights[1][i, :]))
            self.perceptron_weights[1][i, :] /= sum_w_1

    def _predict_one_base(self, X):
        return self.perceptron_weights[0] @ X

    def _predict_one_meta(self, X):
        return self.perceptron_weights[1] @ X


class SSTLearningNodeAdaptive(SSTLearningNode):
    def __init__(self, initial_stats=None, parent_node=None, random_state=None):
        super().__init__(initial_stats, parent_node, random_state)

        # Faded adaptive errors
        self.fMAE_M = 0.0
        self.fMAE_P = 0.0
        # Stacked Perceptron
        self.fMAE_SP = 0.0

    def predict_one(self, X, *, tree=None):
        predictions = self.stats[1] / self.stats[0] if len(self.stats) > 0 else np.zeros(
            tree._n_targets, dtype=np.float64)

        if self.perceptron_weights is None:
            return predictions

        X_norm = tree.normalize_sample(X)

        # Standard perceptron
        Y_pred_base = self._predict_one_base(X_norm)
        # Stacked perceptron
        Y_pred_meta = self._predict_one_meta(np.append(Y_pred_base, 1.0))

        mean = tree.sum_of_values / tree.examples_seen
        variance = ((tree.sum_of_squares - (tree.sum_of_values * tree.sum_of_values)
                    / tree.examples_seen) / (tree.examples_seen - 1))
        sd = np.sqrt(variance, out=np.zeros_like(variance), where=variance >= 0.0)

        pred_P = Y_pred_base * sd + mean
        pred_SP = Y_pred_meta * sd + mean

        # Selects, for each target, the best current performer
        for j in range(tree._n_targets):
            b_pred = np.argmin([self.fMAE_M[j], self.fMAE_P[j], self.fMAE_SP[j]])

            if b_pred == 0:
                # If all the expected errors are the same,
                # use the standard perceptron
                if self.fMAE_M[j] == self.fMAE_P[j] == self.fMAE_SP[j]:
                    predictions[j] = pred_P[j]
            else:
                if b_pred == 1:
                    # Use the stacked perceptron if its expected error is the same than the error
                    # for the standard perceptron
                    if self.fMAE_P[j] == self.fMAE_SP[j]:
                        predictions[j] = pred_SP[j]
                    else:
                        predictions[j] = pred_P[j]
                else:
                    predictions[j] = pred_SP[j]
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
        Y_pred_base = self._predict_one_base(X_norm)

        Y_norm = tree.normalize_target_value(y)

        self.perceptron_weights[0] += learning_ratio * (Y_norm - Y_pred_base)[:, None] \
            @ X_norm[None, :]

        # Add bias term
        Y_pred_base = np.append(Y_pred_base, 1.0)
        Y_pred_meta = self._predict_one_meta(Y_pred_base)

        self.perceptron_weights[1] += learning_ratio * (Y_norm - Y_pred_meta)[:, None] \
            @ Y_pred_base[None, :]

        self._normalize_perceptron_weights()

        # Update faded errors for the predictors
        # The considered errors are normalized, since they are based on mean centered and std
        # scaled values
        self.fMAE_M = (0.95 * self.fMAE_M + np.absolute(Y_norm - tree.normalize_target_value(
            self._stats[1] / self._stats[0])))

        # Ignore added bias term in the comparison
        self.fMAE_P = 0.95 * self.fMAE_P + np.absolute(Y_norm - Y_pred_base[:-1])

        self.fMAE_SP = 0.95 * self.fMAE_SP + np.absolute(Y_norm - Y_pred_meta)


class SSTActiveLearningNode(SSTLearningNode, ActiveLeafRegressor):
    """ Learning Node for SST-HT that always use stacked perceptrons to provide
    targets responses.

    Parameters
    ----------
    initial_stats: dict
        A dictionary containing the set of sufficient statistics to be
        stored by the leaf node. It contains the following elements:
        - 0: the sum of elements seen so far;
        - 1: the sum of the targets values seen so far;
        - 2: the sum of the squared values of the targets seen so far.
    parent_node: SSTActiveLearningNode (default=None)
        A node containing statistics about observed data.
    random_state : `int`, `RandomState` instance or None (default=None)
        If int, `random_state` is used as seed to the random number
        generator; If a `RandomState` instance, `random_state` is the
        random number generator; If `None`, the random number generator
        is the current `RandomState` instance used by `np.random`.
    """
    def __init__(self, initial_stats=None, parent_node=None, random_state=None):
        """ SSTActiveLearningNode class constructor."""
        super().__init__(initial_stats, parent_node, random_state)


class SSTInactiveLearningNode(SSTLearningNode, InactiveLeaf):
    """ Inactive Learning Node for SST-HT that always use stacked perceptrons to
    provide targets responses.

    Parameters
    ----------
    initial_stats: dict
        A dictionary containing the set of sufficient statistics to be
        stored by the leaf node. It contains the following elements:
        - 0: the sum of elements seen so far;
        - 1: the sum of the targets values seen so far;
        - 2: the sum of the squared values of the targets seen so far.
    parent_node: SSTActiveLearningNode (default=None)
        A node containing statistics about observed data.
    random_state : `int`, `RandomState` instance or None (default=None)
        If int, `random_state` is used as seed to the random number
        generator; If a `RandomState` instance, `random_state` is the
        random number generator; If `None`, the random number generator
        is the current `RandomState` instance used by `np.random`.
    """
    def __init__(self, initial_stats=None, parent_node=None, random_state=None):
        """ SSTInactiveLearningNode class constructor."""
        super().__init__(initial_stats, parent_node, random_state)


class SSTActiveLearningNodeAdaptive(SSTLearningNodeAdaptive, ActiveLeafRegressor):
    """ Multi-target regression learning node for SST-HT that keeps track of
    mean, perceptron, and stacked perceptrons predictors for each target.

    Parameters
    ----------
    initial_stats: dict
        A dictionary containing the set of sufficient statistics to be
        stored by the leaf node. It contains the following elements:
        - 0: the sum of elements seen so far;
        - 1: the sum of the targets values seen so far;
        - 2: the sum of the squared values of the targets seen so far.
    parent_node: SSTActiveLearningNodeAdaptive (default=None)
        A node containing statistics about observed data.
    random_state : `int`, `RandomState` instance or None (default=None)
        If int, `random_state` is used as seed to the random number
        generator; If a `RandomState` instance, `random_state` is the
        random number generator; If `None`, the random number generator
        is the current `RandomState` instance used by `np.random`.
    """
    def __init__(self, initial_stats=None, parent_node=None, random_state=None):
        """ SSTActiveLearningNodeAdaptive class constructor. """
        super().__init__(initial_stats, parent_node, random_state)


class SSTInactiveLearningNodeAdaptive(SSTLearningNodeAdaptive, InactiveLeaf):
    """ Inactive Multi-target regression learning node for SST-HT that keeps
    track of mean, perceptron, and stacked perceptrons predictors for each
    target.

    Parameters
    ----------
    initial_stats: dict
        A dictionary containing the set of sufficient statistics to be
        stored by the leaf node. It contains the following elements:
        - 0: the sum of elements seen so far;
        - 1: the sum of the targets values seen so far;
        - 2: the sum of the squared values of the targets seen so far.
    parent_node: SSTActiveLearningNodeAdaptive (default=None)
        A node containing statistics about observed data.
    random_state : `int`, `RandomState` instance or None (default=None)
        If int, `random_state` is used as seed to the random number
        generator; If a `RandomState` instance, `random_state` is the
        random number generator; If `None`, the random number generator
        is the current `RandomState` instance used by `np.random`.
    """
    def __init__(self, initial_stats=None, parent_node=None, random_state=None):
        """ SSTInactiveLearningNodeAdaptive class constructor."""
        super().__init__(initial_stats, parent_node, random_state)
