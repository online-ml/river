import numpy as np

from skmultiflow.trees.nodes import ActiveLearningNodeForRegression
from skmultiflow.trees.attribute_observer import NominalAttributeRegressionObserver
from skmultiflow.trees.attribute_observer import NumericAttributeRegressionObserver

from skmultiflow.utils import check_random_state, get_dimensions


class RandomLearningNodeForRegression(ActiveLearningNodeForRegression):
    """ Learning Node for regression tasks that always use the average target
    value as response.

    Parameters
    ----------
    initial_class_observations: dict
        In regression tasks this dictionary carries the sufficient to perform
        online variance calculation. They refer to the number of observations
        (key '0'), the sum of the target values (key '1'), and the sum of the
        squared target values (key '2').
    max_features: int
        Number of attributes per subset for each node split.
    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    def __init__(self, initial_class_observations, max_features, random_state=None):
        """ ActiveLearningNodeForRegression class constructor. """
        super().__init__(initial_class_observations)

        self.max_features = max_features
        self.list_attributes = np.array([])
        self.random_state = check_random_state(random_state)

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
        ht: HoeffdingTreeRegressor
            Hoeffding Tree to update.

        """
        try:
            self._observed_class_distribution[0] += weight
            self._observed_class_distribution[1] += y * weight
            self._observed_class_distribution[2] += y * y * weight
        except KeyError:
            self._observed_class_distribution[0] = weight
            self._observed_class_distribution[1] = y * weight
            self._observed_class_distribution[2] = y * y * weight

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
