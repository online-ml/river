import numpy as np
from skmultiflow.trees.nodes import InactiveLearningNode


class InactiveLearningNodePerceptron(InactiveLearningNode):
    """ Inactive Learning Node for regression tasks that always use a linear
    perceptron model to provide responses.

    Parameters
    ----------
    initial_class_observations: dict
        In regression tasks this dictionary carries the sufficient to perform
        online variance calculation. They refer to the number of observations
        (key '0'), the sum of the target values (key '1'), and the sum of the
        squared target values (key '2').
    perceptron_weight: np.ndarray(n_features) or None, optional (default=None)
        (default=None)
        The weights for the linear models. If
        not passed, uniform values in the range [-1, 1] are used.
    """
    def __init__(self, initial_class_observations, perceptron_weight=None):
        """ InactiveLearningNodePerceptron class constructor."""
        super().__init__(initial_class_observations)
        if perceptron_weight is None:
            self.perceptron_weight = []
        else:
            self.perceptron_weight = perceptron_weight

    def learn_from_instance(self, X, y, weight, rht):
        """Update the node with the provided instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes for updating the node.
        y: double
            Instance target value.
        weight: float
            Instance weight.
        rht: RegressionHoeffdingTree
            Regression Hoeffding Tree to update.

        """
        if self.perceptron_weight is None:
            self.perceptron_weight = np.random.uniform(-1, 1, len(X)+1)

        try:
            self._observed_class_distribution[0] += weight
        except KeyError:
            self._observed_class_distribution[0] = weight

        if rht.learning_ratio_const:
            learning_ratio = rht.learning_ratio_perceptron
        else:
            learning_ratio = rht.learning_ratio_perceptron / 1 + \
                self._observed_class_distribution[0] * rht.learning_ratio_decay

        try:
            self._observed_class_distribution[1] += y * weight
            self._observed_class_distribution[2] += y * y * weight
        except KeyError:
            self._observed_class_distribution[1] = y * weight
            self._observed_class_distribution[2] = y * y * weight

        for i in range(int(weight)):
            self.update_weights(X, y, learning_ratio, rht)

    def update_weights(self, X, y, learning_ratio, ht):
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
        rht: RegressionHoeffdingTree
            Regression Hoeffding Tree to update.
        """
        normalized_sample = ht.normalize_sample(X)
        normalized_pred = self.predict(normalized_sample)
        normalized_target_value = ht.normalized_target_value(y)
        self.perceptron_weight += learning_ratio * \
            np.multiply((normalized_pred - normalized_target_value),
                        normalized_sample)

    def predict(self, X):
        return np.dot(self.perceptron_weight, X[0])
