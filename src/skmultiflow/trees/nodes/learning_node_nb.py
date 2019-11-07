from skmultiflow.bayes import do_naive_bayes_prediction
from skmultiflow.trees.nodes import ActiveLearningNode


class LearningNodeNB(ActiveLearningNode):
    """ Learning node that uses Naive Bayes models.

    Parameters
    ----------
    initial_class_observations: dict (class_value, weight) or None
        Initial class observations

    """

    def __init__(self, initial_class_observations):
        """ LearningNodeNB class constructor. """
        super().__init__(initial_class_observations)

    def get_class_votes(self, X, ht):
        """ Get the votes per class for a given instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes.
        ht: HoeffdingTree
            Hoeffding Tree.

        Returns
        -------
        dict (class_value, weight)
            Class votes for the given instance.

        """
        if self.get_weight_seen() >= ht.nb_threshold:
            return do_naive_bayes_prediction(
                X, self._observed_class_distribution, self._attribute_observers
            )
        else:
            return super().get_class_votes(X, ht)

    def disable_attribute(self, att_index):
        """ Disable an attribute observer.

        Disabled in Nodes using Naive Bayes, since poor attributes are used in
        Naive Bayes calculation.

        Parameters
        ----------
        att_index: int
            Attribute index.

        """
        pass
