from skmultiflow.trees.nodes import InactiveLearningNode


class InactiveLearningNodeForRegression(InactiveLearningNode):
    """ Inactive Learning Node for regression tasks that always use
    the average target value as response.

    Parameters
    ----------
    initial_class_observations: dict
        In regression tasks this dictionary carries the sufficient to perform
        online variance calculation. They refer to the number of observations
        (key '0'), the sum of the target values (key '1'), and the sum of the
        squared target values (key '2').
    """
    def __init__(self, initial_class_observations):
        """ InactiveLearningNodeForRegression class constructor."""
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
