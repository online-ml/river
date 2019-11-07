from skmultiflow.trees.nodes import ActiveLearningNode
from skmultiflow.trees.attribute_observer import NominalAttributeRegressionObserver
from skmultiflow.trees.attribute_observer import NumericAttributeRegressionObserver


class ActiveLearningNodeForRegression(ActiveLearningNode):
    """ Learning Node for regression tasks that always use the average target
    value as response.

    Parameters
    ----------
    initial_class_observations: dict
        In regression tasks this dictionary carries the sufficient to perform
        online variance calculation. They refer to the number of observations
        (key '0'), the sum of the target values (key '1'), and the sum of the
        squared target values (key '2').
    """
    def __init__(self, initial_class_observations):
        """ ActiveLearningNodeForRegression class constructor. """
        super().__init__(initial_class_observations)

    def learn_from_instance(self, X, y, weight, ht):
        """Update the node with the provided instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes for updating the node.
        y: float
            Instance target value.
        weight: float
            Instance weight.
        ht: HoeffdingTree
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

        for i in range(len(X)):
            try:
                obs = self._attribute_observers[i]
            except KeyError:
                if ht.nominal_attributes is not None and i in ht.nominal_attributes:
                    obs = NominalAttributeRegressionObserver()
                else:
                    obs = NumericAttributeRegressionObserver()
                self._attribute_observers[i] = obs
            obs.observe_attribute_class(X[i], y, weight)

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
