import numpy as np
from skmultiflow.trees.nodes import ActiveLearningNodePerceptronMultiTarget
from skmultiflow.trees.attribute_observer import NumericAttributeRegressionObserverMultiTarget
from skmultiflow.trees.attribute_observer import NominalAttributeRegressionObserver
from skmultiflow.utils import get_dimensions


class SSTActiveLearningNode(ActiveLearningNodePerceptronMultiTarget):
    """ Learning Node for SST-HT that always use stacked perceptrons to provide
    targets responses.

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
        The weight matrix for the perceptron predictors. Set to `None`
        by default (in that case it will be randomly initiated).
    random_state : `int`, `RandomState` instance or None (default=None)
        If int, `random_state` is used as seed to the random number
        generator; If a `RandomState` instance, `random_state` is the
        random number generator; If `None`, the random number generator
        is the current `RandomState` instance used by `np.random`.
    """
    def __init__(self, initial_class_observations, perceptron_weight=None,
                 random_state=None):
        """ SSTActiveLearningNode class constructor."""
        super().__init__(initial_class_observations, perceptron_weight,
                         random_state)

    def learn_from_instance(self, X, y, weight, rht):
        """Update the node with the provided instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes for updating the node.
        y: numpy.ndarray of length equal to the number of targets.
            Instance targets.
        weight: float
            Instance weight.
        rht: RegressionHoeffdingTree
            Regression Hoeffding Tree to update.
        """
        if self.perceptron_weight is None:
            self.perceptron_weight = {}
            # Creates matrix of perceptron random weights
            _, rows = get_dimensions(y)
            _, cols = get_dimensions(X)

            self.perceptron_weight[0] = \
                self.random_state.uniform(-1.0, 1.0, (rows, cols + 1))
            # Cascade Stacking
            self.perceptron_weight[1] = \
                self.random_state.uniform(-1.0, 1.0, (rows, rows + 1))
            self.normalize_perceptron_weights()

        try:
            self._observed_class_distribution[0] += weight
        except KeyError:
            self._observed_class_distribution[0] = weight

        if rht.learning_ratio_const:
            learning_ratio = rht.learning_ratio_perceptron
        else:
            learning_ratio = rht.learning_ratio_perceptron / \
                             (1 + self._observed_class_distribution[0] *
                              rht.learning_ratio_decay)

        try:
            self._observed_class_distribution[1] += weight * y
            self._observed_class_distribution[2] += weight * y * y
        except KeyError:
            self._observed_class_distribution[1] = weight * y
            self._observed_class_distribution[2] = weight * y * y

        for i in range(int(weight)):
            self.update_weights(X, y, learning_ratio, rht)

        for i, x in enumerate(X):
            try:
                obs = self._attribute_observers[i]
            except KeyError:
                # Creates targets observers, if not already defined
                if rht.nominal_attributes is not None and i in rht.nominal_attributes:
                    obs = NominalAttributeRegressionObserver()
                else:
                    obs = NumericAttributeRegressionObserverMultiTarget()
                self._attribute_observers[i] = obs
            obs.observe_attribute_class(x, y, weight)

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

    # Normalize both levels
    def normalize_perceptron_weights(self):
        """ Normalizes both levels of perceptron weights."""
        n_targets = self.perceptron_weight[0].shape[0]
        # Normalize perceptron weights
        for i in range(n_targets):
            sum_w_0 = np.sum(np.absolute(self.perceptron_weight[0][i, :]))
            self.perceptron_weight[0][i, :] /= sum_w_0
            sum_w_1 = np.sum(np.absolute(self.perceptron_weight[1][i, :]))
            self.perceptron_weight[1][i, :] /= sum_w_1

    def _predict_base(self, X):
        return self.perceptron_weight[0] @ X

    def _predict_meta(self, X):
        return self.perceptron_weight[1] @ X

    def get_weight_seen(self):
        """Calculate the total weight seen by the node.

        Returns
        -------
        float
            Total weight seen.
        """
        if self._observed_class_distribution == {}:
            return 0
        else:
            return self._observed_class_distribution[0]
