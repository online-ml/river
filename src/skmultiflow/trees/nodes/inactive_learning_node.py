from skmultiflow.trees.nodes import LearningNode


class InactiveLearningNode(LearningNode):
    """ Inactive learning node that does not grow.

    Parameters
    ----------
    initial_class_observations: dict (class_value, weight) or None
        Initial class observations

    """

    def __init__(self, initial_class_observations=None):
        """ InactiveLearningNode class constructor. """
        super().__init__(initial_class_observations)

    def learn_from_instance(self, X, y, weight, ht):
        """ Update the node with the provided instance.

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
            self._observed_class_distribution = dict(sorted(self._observed_class_distribution.items()))
