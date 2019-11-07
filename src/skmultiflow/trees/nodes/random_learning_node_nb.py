from skmultiflow.bayes import do_naive_bayes_prediction
from skmultiflow.trees.nodes import RandomLearningNodeClassification


class RandomLearningNodeNB(RandomLearningNodeClassification):
    """ARF Naive Bayes learning node class.

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

    def __init__(self, initial_class_observations, max_features, random_state):
        """ LearningNodeNB class constructor. """
        super().__init__(initial_class_observations, max_features, random_state)

    def get_class_votes(self, X, ht):
        """Get the votes per class for a given instance.

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
