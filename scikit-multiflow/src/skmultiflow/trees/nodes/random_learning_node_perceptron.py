import numpy as np

from skmultiflow.trees.nodes import ActiveLearningNodePerceptron
from skmultiflow.trees.attribute_observer import NominalAttributeRegressionObserver
from skmultiflow.trees.attribute_observer import NumericAttributeRegressionObserver
from skmultiflow.utils import get_dimensions


class RandomLearningNodePerceptron(ActiveLearningNodePerceptron):
    """ Learning Node for regression tasks that always use a linear perceptron
    model to provide responses.

    Parameters
    ----------
    initial_class_observations: dict
        In regression tasks this dictionary carries the sufficient statistics
        to perform online variance calculation. They refer to the number of
        observations (key '0'), the sum of the target values (key '1'), and
        the sum of the squared target values (key '2').
    max_features: int
        Number of attributes per subset for each node split.
    parent_node: RandomLearningNodePerceptron (default=None)
        A node containing statistics about observed data.
    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """

    def __init__(self, initial_class_observations, max_features, parent_node=None,
                 random_state=None):
        super().__init__(initial_class_observations, parent_node, random_state)
        self.max_features = max_features
        self.list_attributes = np.array([])

    def learn_from_instance(self, X, y, weight, rht):
        """Update the node with the provided instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes for updating the node.
        y: float
            Instance target value.
        weight: float
            Instance weight.
        rht: HoeffdingTreeRegressor
            Regression Hoeffding Tree to update.

        """

        # In regression, the self._observed_class_distribution dictionary keeps three statistics:
        # [0] sum of sample seen by the node
        # [1] sum of target values
        # [2] sum of squared target values
        # These statistics are useful to calculate the mean and to calculate the variance reduction

        if self.perceptron_weight is None:
            self.perceptron_weight = self.random_state.uniform(-1, 1, len(X)+1)

        try:
            self._observed_class_distribution[0] += weight
            self._observed_class_distribution[1] += y * weight
            self._observed_class_distribution[2] += y * y * weight
        except KeyError:
            self._observed_class_distribution[0] = weight
            self._observed_class_distribution[1] = y * weight
            self._observed_class_distribution[2] = y * y * weight

        # Update perceptron
        self.samples_seen = self._observed_class_distribution[0]

        if rht.learning_ratio_const:
            learning_ratio = rht.learning_ratio_perceptron
        else:
            learning_ratio = rht.learning_ratio_perceptron / \
                             (1 + self.samples_seen * rht.learning_ratio_decay)

        # Loop for compatibility with bagging methods
        for i in range(int(weight)):
            self.update_weights(X, y, learning_ratio, rht)

        if self.list_attributes.size == 0:
            self.list_attributes = self._sample_features(get_dimensions(X)[1])

        for i in self.list_attributes:
            try:
                obs = self._attribute_observers[i]
            except KeyError:
                if rht.nominal_attributes is not None and i in rht.nominal_attributes:
                    obs = NominalAttributeRegressionObserver()
                else:
                    obs = NumericAttributeRegressionObserver()
                self._attribute_observers[i] = obs
            obs.observe_attribute_class(X[i], y, weight)

    def _sample_features(self, n_features):
        return self.random_state.choice(
            n_features, size=self.max_features, replace=False
        )
