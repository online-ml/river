import numpy as np
from skmultiflow.trees.attribute_split_suggestion import AttributeSplitSuggestion
from skmultiflow.trees.nodes import ActiveLearningNode


class AnyTimeActiveLearningNode(ActiveLearningNode):
    """ Active Learning node for the Hoeffding Anytime Tree.

    Parameters
    ----------
    initial_class_observations: dict (class_value, weight) or None
        Initial class observations

    """

    def __init__(self, initial_class_observations):
        """ AnyTimeActiveLearningNode class constructor. """
        super().__init__(initial_class_observations)

    # Override
    def get_best_split_suggestions(self, criterion, ht):
        """ Find possible split candidates without taking into account the the
        null split.

        Parameters
        ----------
        criterion: SplitCriterion
            The splitting criterion to be used.
        ht: HoeffdingTree
            Hoeffding Tree.

        Returns
        -------
        list
            Split candidates.

        """
        best_suggestions = []
        pre_split_dist = self._observed_class_distribution

        for i, obs in self._attribute_observers.items():
            best_suggestion = obs.get_best_evaluated_split_suggestion(
                criterion, pre_split_dist, i, ht.binary_split
            )
            if best_suggestion is not None:
                best_suggestions.append(best_suggestion)

        return best_suggestions

    def get_null_split(self, criterion):
        """ Compute the null split (don't split).

        Parameters
        ----------
        criterion: SplitCriterion
            The splitting criterion to be used.


        Returns
        -------
        list
            Split candidates.

        """

        pre_split_dist = self._observed_class_distribution

        null_split = AttributeSplitSuggestion(
            None, [{}], criterion.get_merit_of_split(pre_split_dist, [pre_split_dist])
        )
        return null_split

    @staticmethod
    def count_nodes():
        """ Calculate the number of split node and leaf starting from this node
        as a root.

        Returns
        -------
        list[int int]
            [number of split node, number of leaf node].

        """
        return np.array([0, 1])
