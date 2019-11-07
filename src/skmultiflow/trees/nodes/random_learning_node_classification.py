import numpy as np
from skmultiflow.utils import get_dimensions

from skmultiflow.trees.nodes import ActiveLearningNode
from skmultiflow.trees.attribute_observer import NominalAttributeClassObserver
from skmultiflow.trees.attribute_observer import NumericAttributeClassObserverGaussian


class RandomLearningNodeClassification(ActiveLearningNode):
    """ARF learning node class.

    Parameters
    ----------
    initial_class_observations: dict (class_value, weight) or None
        Initial class observations.

    max_features: int
        Number of attributes per subset for each node split.

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    def __init__(self, initial_class_observations, max_features, random_state=None):
        """ RandomLearningNodeClassification class constructor. """
        super().__init__(initial_class_observations)

        self.max_features = max_features
        self._attribute_observers = {}
        self.list_attributes = np.array([])
        self.random_state = random_state

    def learn_from_instance(self, X, y, weight, ht):
        """Update the node with the provided instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes for updating the node.
        y: int
            Instance class.
        weight: float
            Instance weight.
        ht: HoeffdingTree
            Hoeffding Tree to update.
        """
        try:
            self._observed_class_distribution[y] += weight
        except KeyError:
            self._observed_class_distribution[y] = weight
            self._observed_class_distribution = dict(
                sorted(self._observed_class_distribution.items())
            )

        if self.list_attributes.size == 0:
            self.list_attributes = self._sample_features(get_dimensions(X)[1])

        for i in self.list_attributes:
            try:
                obs = self._attribute_observers[i]
            except KeyError:
                if ht.nominal_attributes is not None and i in ht.nominal_attributes:
                    obs = NominalAttributeClassObserver()
                else:
                    obs = NumericAttributeClassObserverGaussian()
                self._attribute_observers[i] = obs
            obs.observe_attribute_class(X[i], int(y), weight)

    def _sample_features(self, n_features):
        return self.random_state.choice(
            n_features, size=self.max_features, replace=False
        )
