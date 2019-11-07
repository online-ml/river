import numpy as np
from skmultiflow.trees.attribute_split_suggestion import AttributeSplitSuggestion
from skmultiflow.trees.attribute_observer import NominalAttributeClassObserver
from skmultiflow.trees.attribute_observer import NumericAttributeClassObserverGaussian
from skmultiflow.trees.nodes import SplitNode


class AnyTimeSplitNode(SplitNode):
    """ Node that splits the data in a Hoeffding Anytime Tree.

    Parameters
    ----------
    split_test: InstanceConditionalTest
        Split test.
    class_observations: dict (class_value, weight) or None
        Class observations
    attribute_observers : dict (attribute id, AttributeClassObserver)
        Attribute Observers
    """

    def __init__(self, split_test, class_observations, attribute_observers):
        """ AnyTimeSplitNode class constructor."""
        super().__init__(split_test, class_observations)
        self._attribute_observers = attribute_observers
        self._weight_seen_at_last_split_reevaluation = 0

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

    @staticmethod
    def find_attribute(id_att, split_suggestions):
        """ Find the attribute given the id.

        Parameters
        ----------
        id_att: int.
            Id of attribute to find.
        split_suggestions: list
            Possible split candidates.
        Returns
        -------
        AttributeSplitSuggestion
            Found attribute.
        """

        # return current attribute as AttributeSplitSuggestion
        x_current = None
        for attSplit in split_suggestions:
            selected_id = attSplit.split_test.get_atts_test_depends_on()[0]
            if selected_id == id_att:
                x_current = attSplit

        return x_current

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
        # Update attribute_observers
        try:
            self._observed_class_distribution[y] += weight
        except KeyError:
            self._observed_class_distribution[y] = weight

        for i in range(len(X)):
            try:
                obs = self._attribute_observers[i]
            except KeyError:
                if i in ht.nominal_attributes:
                    obs = NominalAttributeClassObserver()
                else:
                    obs = NumericAttributeClassObserverGaussian()
                self._attribute_observers[i] = obs
            obs.observe_attribute_class(X[i], int(y), weight)

    def get_weight_seen(self):
        """ Calculate the total weight seen by the node.

        Returns
        -------
        float
            Total weight seen.

        """
        return sum(self._observed_class_distribution.values())

    def get_attribute_observers(self):
        """ Get attribute observers at this node.

        Returns
        -------
        dict (attribute id, AttributeClassObserver)
            Attribute observers of this node.

        """
        return self._attribute_observers

    def get_weight_seen_at_last_split_reevaluation(self):
        """ Get the weight seen at the last split reevaluation.

        Returns
        -------
        float
            Total weight seen at last split reevaluation.

        """
        return self._weight_seen_at_last_split_reevaluation

    def update_weight_seen_at_last_split_reevaluation(self):
        """ Update weight seen at the last split in the reevaluation. """
        self._weight_seen_at_last_split_reevaluation = sum(
            self._observed_class_distribution.values()
        )

    def count_nodes(self):
        """ Calculate the number of split node and leaf starting from this node
        as a root.

        Returns
        -------
        list[int int]
            [number of split node, number of leaf node].

        """

        count = np.array([1, 0])
        # get children
        for branch_idx in range(self.num_children()):
            child = self.get_child(branch_idx)
            if child is not None:
                count += child.count_nodes()

        return count
