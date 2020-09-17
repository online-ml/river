import numpy as np

from skmultiflow.trees._attribute_test import AttributeSplitSuggestion

from .base import SplitNode
from .htc_nodes import ActiveLeafClass
from .htc_nodes import LearningNodeMC, LearningNodeNB, LearningNodeNBA
from .htc_nodes import InactiveLearningNodeMC


class EFDTActiveLeaf(ActiveLeafClass):
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

        pre_split_dist = self.stats
        null_split = AttributeSplitSuggestion(
            None, [{}], criterion.get_merit_of_split(pre_split_dist, [pre_split_dist])
        )
        # Force null slot merit to be 0 instead of -infinity
        if null_split.merit == -np.inf:
            null_split.merit = 0.0

        return null_split

    def get_best_split_suggestions(self, criterion, tree):
        """ Find possible split candidates without taking into account the the
        null split.

        Parameters
        ----------
        criterion: SplitCriterion
            The splitting criterion to be used.
        ht: HoeffdingTreeClassifier
            Hoeffding Tree.

        Returns
        -------
        list
            Split candidates.

        """

        best_suggestions = []
        pre_split_dist = self.stats

        for idx, obs in self.attribute_observers.items():
            best_suggestion = obs.get_best_evaluated_split_suggestion(
                criterion, pre_split_dist, idx, tree.binary_split
            )
            if best_suggestion is not None:
                best_suggestions.append(best_suggestion)

        return best_suggestions

    def count_nodes(self):
        """ Calculate the number of split node and leaf starting from this node
        as a root.

        Returns
        -------
        list[int int]
            [number of split node, number of leaf node].

        """
        return np.array([0, 1])


class EFDTSplitNode(SplitNode, EFDTActiveLeaf):
    """ Node that splits the data in a Hoeffding Anytime Tree.

    Parameters
    ----------
    split_test: InstanceConditionalTest
        Split test.
    stats: dict (class_value, weight) or None
        Class observations
    attribute_observers : dict (attribute id, AttributeObserver)
        Attribute Observers
    """

    def __init__(self, split_test, stats, attribute_observers):
        """ AnyTimeSplitNode class constructor."""
        super().__init__(split_test, stats)  # Calls split node constructor
        self.attribute_observers = attribute_observers
        self._weight_seen_at_last_split_reevaluation = 0

    def update_stats(self, y, weight):
        try:
            self.stats[y] += weight
        except KeyError:
            self.stats[y] = weight
            self.stats = dict(sorted(self.stats.items()))

    def learn_one(self, X, y, *, weight=1.0, tree=None):
        """Update the node with the provided sample.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Sample attributes for updating the node.
        y: int or float
            Target value.
        weight: float
            Sample weight.
        tree:
            Tree to update.

        """
        y = int(y)
        self.update_stats(y, weight)
        self.update_attribute_observers(X, y, weight, tree)

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
        self._weight_seen_at_last_split_reevaluation = self.total_weight

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
        for branch_idx in range(self.n_children):
            child = self.get_child(branch_idx)
            if child is not None:
                count += child.count_nodes()

        return count

    @property
    def total_weight(self):
        """ Calculate the total weight seen by the node.

        Returns
        -------
        float
            Total weight seen.

        """
        return sum(self.stats.values())

    def observed_class_distribution_is_pure(self):
        """ Check if observed class distribution is pure, i.e. if all samples
        belong to the same class.

        Returns
        -------
        boolean
            True if observed number of classes is less than 2, False otherwise.

        """
        count = 0
        for _, weight in self._stats.items():
            if weight != 0:
                count += 1
                if count == 2:  # No need to count beyond this point
                    break
        return count < 2


class EFDTActiveLearningNodeMC(LearningNodeMC, EFDTActiveLeaf):
    """ Active Learning node for the Hoeffding Anytime Tree.

    Parameters
    ----------
    initial_stats: dict (class_value, weight) or None
        Initial class observations

    """

    def __init__(self, initial_stats=None):
        """ AnyTimeActiveLearningNode class constructor. """
        super().__init__(initial_stats)


class EFDTInactiveLearningNodeMC(InactiveLearningNodeMC):
    """ Inactive Learning node for the Hoeffding Anytime Tree.

    Parameters
    ----------
    initial_stats: dict (class_value, weight) or None
        Initial class observations

    """

    def __init__(self, initial_stats=None):
        """ InactiveLearningNode class constructor. """
        super().__init__(initial_stats)

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


class EFDTActiveLearningNodeNB(LearningNodeNB, EFDTActiveLeaf):
    """ Learning node  for the Hoeffding Anytime Tree that uses Naive Bayes
    models.

    Parameters
    ----------
    initial_stats: dict (class_value, weight) or None
        Initial class observations

    """
    def __init__(self, initial_stats=None):
        """ EFDTActiveLearningNodeNB class constructor. """
        super().__init__(initial_stats)

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


class EFDTActiveLearningNodeNBA(LearningNodeNBA, EFDTActiveLeaf):
    """ Learning node for the Hoeffding Anytime Tree that uses Adaptive Naive
    Bayes models.

    Parameters
    ----------
    initial_stats: dict (class_value, weight) or None
        Initial class observations

    """
    def __init__(self, initial_stats=None):
        """ AnyTimeLearningNodeNBAdaptive class constructor. """
        super().__init__(initial_stats)

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
