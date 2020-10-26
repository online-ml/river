from collections import Counter
import math
import numbers

from river.tree._attribute_test import AttributeSplitSuggestion
from river.tree._attribute_observer import NominalAttributeClassObserver
from river.tree._attribute_observer import NumericAttributeClassObserverGaussian

from .base import SplitNode
from .htc_nodes import LearningNodeMC, LearningNodeNB, LearningNodeNBA


class EFDTLearningNodeMC(LearningNodeMC):
    """Learning node for the Hoeffding Anytime Tree that always use the majority class."""
    def null_split(self, criterion):
        """Compute the null split (don't split).

        Parameters
        ----------
        criterion
            The splitting criterion to be used.

        Returns
        -------
            The null split candidate.
        """
        pre_split_dist = self.stats
        null_split = AttributeSplitSuggestion(
            None, [{}], criterion.merit_of_split(pre_split_dist, [pre_split_dist])
        )
        # Force null slot merit to be 0 instead of -infinity
        if math.isinf(null_split.merit):
            null_split.merit = 0.0

        return null_split

    def best_split_suggestions(self, criterion, tree):
        """Find possible split candidates without taking into account the
        null split.

        Parameters
        ----------
        criterion
            The splitting criterion to be used.
        tree
            The EFDT which the node belongs to.

        Returns
        -------
            The list of split candidates.
        """
        best_suggestions = []
        pre_split_dist = self.stats

        for idx, obs in self.attribute_observers.items():
            best_suggestion = obs.best_evaluated_split_suggestion(
                criterion, pre_split_dist, idx, tree.binary_split
            )
            if best_suggestion is not None:
                best_suggestions.append(best_suggestion)

        return best_suggestions

    @staticmethod
    def count_nodes():
        """Calculate the number of split node and leaf starting from this node
        as a root.

        Returns
        -------
            A Counter with the number of `leaf_nodes` and `decision_nodes`.
        """
        return Counter(leaf_nodes=1, decision_nodes=0)


class EFDTLearningNodeNB(LearningNodeNB):
    """Learning node  for the Hoeffding Anytime Tree that uses Naive Bayes
    models."""

    def null_split(self, criterion):
        """Compute the null split (don't split).

        Parameters
        ----------
        criterion
            The splitting criterion to be used.

        Returns
        -------
            The null split candidate.
        """
        pre_split_dist = self.stats
        null_split = AttributeSplitSuggestion(
            None, [{}], criterion.merit_of_split(pre_split_dist, [pre_split_dist])
        )
        # Force null slot merit to be 0 instead of -infinity
        if math.isinf(null_split.merit):
            null_split.merit = 0.0

        return null_split

    def best_split_suggestions(self, criterion, tree):
        """Find possible split candidates without taking into account the
        null split.

        Parameters
        ----------
        criterion
            The splitting criterion to be used.
        tree
            The EFDT which the node belongs to.

        Returns
        -------
            The list of split candidates.
        """
        best_suggestions = []
        pre_split_dist = self.stats

        for idx, obs in self.attribute_observers.items():
            best_suggestion = obs.best_evaluated_split_suggestion(
                criterion, pre_split_dist, idx, tree.binary_split
            )
            if best_suggestion is not None:
                best_suggestions.append(best_suggestion)

        return best_suggestions

    @staticmethod
    def count_nodes():
        """Calculate the number of split node and leaf starting from this node
        as a root.

        Returns
        -------
            A Counter with the number of `leaf_nodes` and `decision_nodes`.
        """
        return Counter(leaf_nodes=1, decision_nodes=0)


class EFDTLearningNodeNBA(LearningNodeNBA):
    """Learning node for the Hoeffding Anytime Tree that uses Adaptive Naive
    Bayes models."""

    def null_split(self, criterion):
        """Compute the null split (don't split).

        Parameters
        ----------
        criterion
            The splitting criterion to be used.

        Returns
        -------
            The null split candidate.
        """
        pre_split_dist = self.stats
        null_split = AttributeSplitSuggestion(
            None, [{}], criterion.merit_of_split(pre_split_dist, [pre_split_dist])
        )
        # Force null slot merit to be 0 instead of -infinity
        if math.isinf(null_split.merit):
            null_split.merit = 0.0

        return null_split

    def best_split_suggestions(self, criterion, tree):
        """Find possible split candidates without taking into account the
        null split.

        Parameters
        ----------
        criterion
            The splitting criterion to be used.
        tree
            The EFDT which the node belongs to.

        Returns
        -------
            The list of split candidates.
        """
        best_suggestions = []
        pre_split_dist = self.stats

        for idx, obs in self.attribute_observers.items():
            best_suggestion = obs.best_evaluated_split_suggestion(
                criterion, pre_split_dist, idx, tree.binary_split
            )
            if best_suggestion is not None:
                best_suggestions.append(best_suggestion)

        return best_suggestions

    @staticmethod
    def count_nodes():
        """Calculate the number of split node and leaf starting from this node
        as a root.

        Returns
        -------
            A Counter with the number of `leaf_nodes` and `decision_nodes`.
        """
        return Counter(leaf_nodes=1, decision_nodes=0)


class EFDTSplitNode(SplitNode):
    """Node that splits the data in a EFDT and acts like a learning node (leaf).

    Parameters
    ----------
    split_test
        Split test.
    stats
        Class observations
    depth
        The depth of the node in the tree.
    attribute_observers
        Attribute Observers
    """
    def __init__(self, split_test, stats, depth, attribute_observers):
        super().__init__(split_test, stats, depth)  # Calls split node constructor
        self._attribute_observers = attribute_observers
        self._last_split_reevaluation_at = 0

    @staticmethod
    def new_nominal_attribute_observer(**kwargs):
        return NominalAttributeClassObserver(**kwargs)

    @staticmethod
    def new_numeric_attribute_observer(**kwargs):
        return NumericAttributeClassObserverGaussian(**kwargs)

    @property
    def attribute_observers(self):
        return self._attribute_observers

    @attribute_observers.setter
    def attribute_observers(self, attr_obs):
        self._attribute_observers = attr_obs

    def update_stats(self, y, sample_weight):
        try:
            self.stats[y] += sample_weight
        except KeyError:
            self.stats[y] = sample_weight

    def update_attribute_observers(self, x, y, sample_weight, tree, **kwargs):
        for attr_idx, attr_val in x.items():
            try:
                obs = self.attribute_observers[attr_idx]
            except KeyError:
                if ((tree.nominal_attributes is not None and attr_idx in tree.nominal_attributes)
                        or not isinstance(attr_val, numbers.Number)):
                    obs = self.new_nominal_attribute_observer(**kwargs)
                else:
                    obs = self.new_numeric_attribute_observer(**kwargs)
                self.attribute_observers[attr_idx] = obs
            obs.update(attr_val, y, sample_weight)

    def learn_one(self, x, y, *, sample_weight=1.0, tree=None):
        """Learn from the provided sample.

        Parameters
        ----------
        x
            Sample attributes for updating the node.
        y
            Target value.
        sample_weight
            Sample weight.
        tree
            Tree to update.
        """
        self.update_stats(y, sample_weight)
        self.update_attribute_observers(x, y, sample_weight, tree)

    @staticmethod
    def find_attribute(id_att, split_suggestions):
        """Find the attribute given the id.

        Parameters
        ----------
        id_att
            Id of attribute to find.
        split_suggestions
            Possible split candidates.
        Returns
        -------
            Found attribute.
        """
        # TODO verify the possibility of using dictionaries to go from O(m) to O(1)
        x_current = None
        for att_split in split_suggestions:
            selected_id = att_split.split_test.attrs_test_depends_on()[0]
            if selected_id == id_att:
                x_current = att_split
                break

        return x_current

    @property
    def last_split_reevaluation_at(self) -> float:
        """Get the weight seen at the last split reevaluation.

        Returns
        -------
            Total weight seen at last split reevaluation.
        """
        return self._last_split_reevaluation_at

    @last_split_reevaluation_at.setter
    def last_split_reevaluation_at(self, value: float):
        """Update weight seen at the last split in the reevaluation. """
        self._last_split_reevaluation_at = value

    def count_nodes(self):
        """Calculate the number of split node and leaf starting from this node
        as a root.

        Returns
        -------
            A Counter with the number of `leaf_nodes` and `decision_nodes`.
        """

        count = Counter(leaf_nodes=0, decision_nodes=1)
        # get children
        for branch_idx in range(self.n_children):
            child = self.get_child(branch_idx)
            if child is not None:
                count += child.count_nodes()

        return count

    @property
    def total_weight(self):
        """Calculate the total weight seen by the node.

        Returns
        -------
        float
            Total weight seen.
        """
        return sum(self.stats.values())

    def observed_class_distribution_is_pure(self):
        """Check if observed class distribution is pure, i.e. if all samples
        belong to the same class.

        Returns
        -------
        boolean
            True if observed number of classes is smaller than 2, False otherwise.
        """
        count = 0
        for _, weight in self._stats.items():
            if weight != 0:
                count += 1
                if count == 2:  # No need to count beyond this point
                    break
        return count < 2

    def null_split(self, criterion):
        """Compute the null split (don't split).

        Parameters
        ----------
        criterion
            The splitting criterion to be used.

        Returns
        -------
            The null split candidate.
        """
        pre_split_dist = self.stats
        null_split = AttributeSplitSuggestion(
            None, [{}], criterion.merit_of_split(pre_split_dist, [pre_split_dist])
        )
        # Force null slot merit to be 0 instead of -infinity
        if math.isinf(null_split.merit):
            null_split.merit = 0.0

        return null_split

    def best_split_suggestions(self, criterion, tree):
        """Find possible split candidates without taking into account the
        null split.

        Parameters
        ----------
        criterion
            The splitting criterion to be used.
        tree
            The EFDT which the node belongs to.

        Returns
        -------
            The list of split candidates.
        """
        best_suggestions = []
        pre_split_dist = self.stats

        for idx, obs in self.attribute_observers.items():
            best_suggestion = obs.best_evaluated_split_suggestion(
                criterion, pre_split_dist, idx, tree.binary_split
            )
            if best_suggestion is not None:
                best_suggestions.append(best_suggestion)

        return best_suggestions
