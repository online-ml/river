from skmultiflow.bayes import do_naive_bayes_prediction
from skmultiflow.trees.nodes import LearningNodeNB


class LearningNodeNBAdaptive(LearningNodeNB):
    """ Learning node that uses Adaptive Naive Bayes models.

    Parameters
    ----------
    initial_class_observations: dict (class_value, weight) or None
        Initial class observations

    """

    def __init__(self, initial_class_observations):
        """ LearningNodeNBAdaptive class constructor. """
        super().__init__(initial_class_observations)
        self._mc_correct_weight = 0.0
        self._nb_correct_weight = 0.0

    def learn_from_instance(self, X, y, weight, ht):
        """ Update the node with the provided instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes for updating the node.
        y: int
            Instance class.
        weight: float
            The instance's weight.
        ht: HoeffdingTree
            The Hoeffding Tree to update.

        """
        if self._observed_class_distribution == {}:
            # All classes equal, default to class 0
            if 0 == y:
                self._mc_correct_weight += weight
        elif max(self._observed_class_distribution, key=self._observed_class_distribution.get) == y:
            self._mc_correct_weight += weight
        nb_prediction = do_naive_bayes_prediction(
            X, self._observed_class_distribution, self._attribute_observers
        )
        if max(nb_prediction, key=nb_prediction.get) == y:
            self._nb_correct_weight += weight
        super().learn_from_instance(X, y, weight, ht)

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
        if self._mc_correct_weight > self._nb_correct_weight:
            return self._observed_class_distribution
        return do_naive_bayes_prediction(
            X, self._observed_class_distribution, self._attribute_observers
        )
