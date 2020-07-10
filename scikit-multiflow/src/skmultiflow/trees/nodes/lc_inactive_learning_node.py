from skmultiflow.trees.nodes import InactiveLearningNode


class LCInactiveLearningNode(InactiveLearningNode):
    """ Inactive Learning node for the Label Combination Hoeffding Tree.

    Parameters
    ----------
    initial_class_observations: dict (class_value, weight) or None
        Initial class observations

    """
    def __init__(self, initial_class_observations=None):
        """ LCInactiveLearningNode class constructor. """
        super().__init__(initial_class_observations)

    def learn_from_instance(self, X, y, weight, ht):

        i = ''.join(str(e) for e in y)
        i = int(i, 2)
        try:
            self._observed_class_distribution[i] += weight
        except KeyError:
            self._observed_class_distribution[i] = weight
