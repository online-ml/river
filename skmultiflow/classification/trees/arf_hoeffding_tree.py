__author__ = 'Anderson Carlos Ferreira da Silva'

from random import randint
from skmultiflow.classification.trees.hoeffding_tree import *


class ARFHoeffdingTree(HoeffdingTree):
    """ Adaptive Random Forest Hoeffding Tree.

    Parameters
    __________
    TODO

    Notes
    _____
    This is the base model for the Adaptive Random Forest ensemble learner
    (See skmultiflow.classification.meta.adaptive_random_forests).
    This Hoeffding Tree includes a subspace size parameter, which defines the number of randomly selected features to
    be considered at each split.

    """
    class RandomLearningNode(HoeffdingTree.ActiveLearningNode):
        """Random learning node.

        Parameters
        ----------
        initial_class_observations: dict (class_value, weight) or None
            Initial class observations
        """

        def __init__(self,
                     initial_class_observations,
                     nb_attributes):
            super().__init__(initial_class_observations)
            self.nb_attributes = nb_attributes
            self._attribute_observers = [None] * nb_attributes
            self.list_attributes = []

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
            if y not in self._observed_class_distribution:
                self._observed_class_distribution[y] = 0.0

            self._observed_class_distribution[y] += weight
            if not self.list_attributes:
                self.list_attributes = [None] * self.nb_attributes
                for j in range(self.nb_attributes):
                    is_unique = False
                    while not is_unique:
                        self.list_attributes[j] = randint(0, self.nb_attributes - 1)
                        is_unique = True
                        for i in range(j):
                            if self.list_attributes[j] == self.list_attributes[i]:
                                is_unique = False
                                break

            for j in range(self.nb_attributes):
                i = self.list_attributes[j]
                obs = self._attribute_observers[i]
                if obs is None:
                    if i in ht.nominal_attributes:
                        obs = NominalAttributeClassObserver()
                    else:
                        obs = GaussianNumericAttributeClassObserver()
                    self._attribute_observers[i] = obs
                obs.observe_attribute_class(X[i], int(y), weight)

    class LearningNodeNB(RandomLearningNode):

        def __init__(self, initial_class_observations, nb_attributes):
            super().__init__(initial_class_observations, nb_attributes)

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
                return do_naive_bayes_prediction(X, self._observed_class_distribution, self._attribute_observers)
            else:
                return super().get_class_votes(X, ht)

    class LearningNodeNBAdaptive(LearningNodeNB):
        """Learning node that uses Adaptive Naive Bayes models.
        Parameters
        ----------
        initial_class_observations: dict (class_value, weight) or None
            Initial class observations
        """

        def __init__(self, initial_class_observations, nb_attributes):
            """LearningNodeNBAdaptive class constructor. """
            super().__init__(initial_class_observations, nb_attributes)
            self._mc_correct_weight = 0.0
            self._nb_correct_weight = 0.0

        def learn_from_instance(self, X, y, weight, ht):
            """Update the node with the provided instance.
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
            nb_prediction = do_naive_bayes_prediction(X, self._observed_class_distribution, self._attribute_observers)
            if max(nb_prediction, key=nb_prediction.get) == y:
                self._nb_correct_weight += weight
            super().learn_from_instance(X, y, weight, ht)

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
            if self._mc_correct_weight > self._nb_correct_weight:
                return self._observed_class_distribution
            return do_naive_bayes_prediction(X, self._observed_class_distribution, self._attribute_observers)

    def __init__(self, max_byte_size=33554432, memory_estimate_period=1000000, grace_period=200,
                 split_criterion='info_gain', split_confidence=0.0000001, tie_threshold=0.05,
                 binary_split=False, stop_mem_management=False, remove_poor_atts=False, no_preprune=False,
                 leaf_prediction='nba', nb_threshold=0, nominal_attributes=None, nb_attributes=2):
        """ADFHoeffdingTree class constructor."""
        super().__init__(max_byte_size, memory_estimate_period, grace_period, split_criterion, split_confidence,
                         tie_threshold, binary_split, stop_mem_management, remove_poor_atts, no_preprune,
                         leaf_prediction, nb_threshold, nominal_attributes)
        self.nb_attributes = nb_attributes
        self.remove_poor_attributes_option = None

    def _new_learning_node(self, initial_class_observations=None):
        """Create a new learning node. The type of learning node depends on the tree configuration."""
        if initial_class_observations is None:
            initial_class_observations = {}
        if self._leaf_prediction == MAJORITY_CLASS:
            return self.RandomLearningNode(initial_class_observations, self.nb_attributes)
        elif self._leaf_prediction == NAIVE_BAYES:
            return self.LearningNodeNB(initial_class_observations, self.nb_attributes)
        else:  # NAIVE_BAYES_ADAPTIVE
            return self.LearningNodeNBAdaptive(initial_class_observations, self.nb_attributes)

    @staticmethod
    def is_randomizable():
        return True

    def copy(self):
        return ARFHoeffdingTree(nominal_attributes=self.nominal_attributes, nb_attributes=self.nb_attributes)
