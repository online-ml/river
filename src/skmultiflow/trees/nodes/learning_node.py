from skmultiflow.trees.nodes import Node


class LearningNode(Node):
    """ Base class for Learning Nodes in a Hoeffding Tree.

    Parameters
    ----------
    initial_class_observations: dict (class_value, weight) or None
        Initial class observations

    """

    def __init__(self, initial_class_observations=None):
        """ LearningNode class constructor. """
        super().__init__(initial_class_observations)

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
        pass
