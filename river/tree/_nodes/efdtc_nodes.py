import math
from collections import Counter

from river.utils.skmultiflow_utils import normalize_values_in_dict

from .._attribute_test import SplitSuggestion
from ..splitter.nominal_splitter_classif import NominalSplitterClassif
from .base import SplitNode
from .htc_nodes import LearningNode, LearningNodeMC, LearningNodeNB, LearningNodeNBA


class BaseEFDTNode(LearningNode):
    """Helper class that define basic operations of EFDT's nodes.

    It inherits from `LearningNode` and provides extra functionalities, while changing
    the splitting behavior of its parent class. This is an abstract class, since it does
    not implement all the inherited abstract methods from its parent class. BaseEDFTNode
    is designed to work with other learning/split nodes.

    Parameters
    ----------
    stats
        Class observations.
    depth
        The depth of the node in the tree.
    splitter
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    kwargs
        To ensure compatibility with split nodes.

    Notes
    -----
    The constructor method receives additional kwargs params to ensure it plays nice with
    the multiple inheritance used in the split node of EFDT.
    """

    def __init__(self, stats, depth, splitter, **kwargs):
        super().__init__(stats=stats, depth=depth, splitter=splitter, **kwargs)

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
        null_split = SplitSuggestion(
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

        for idx, splitter in self.splitters.items():
            best_suggestion = splitter.best_evaluated_split_suggestion(
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


class EFDTSplitNode(SplitNode, BaseEFDTNode):
    """Node that splits the data in a EFDT.

    This node is an exception among the tree's nodes. EFDTSplitNode is both a split node
    and a learning node. EFDT updates all of the nodes in the path from the root to a leaf
    when a new instance arrives. Besides that, it also revisit split decisions from time
    to time. For that reason, this decision node also needs to able to learn from new
    instances.

    Parameters
    ----------
    split_test
        Split test.
    stats
        Class observations
    depth
        The depth of the node in the tree.
    splitter
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    existing_splitters
        Existing attribute observers from previous nodes passed to provide a warm-start.
    kwargs
        Other parameters passed to the learning nodes.
    """

    def __init__(
        self, split_test, stats, depth, splitter, existing_splitters, **kwargs
    ):
        super().__init__(
            stats=stats, depth=depth, splitter=splitter, split_test=split_test, **kwargs
        )
        self.splitters = existing_splitters
        self._last_split_reevaluation_at = 0

    @property
    def total_weight(self) -> float:
        return sum(self.stats.values()) if self.stats else 0

    @staticmethod
    def new_nominal_splitter():
        return NominalSplitterClassif()

    def update_stats(self, y, sample_weight):
        try:
            self.stats[y] += sample_weight
        except KeyError:
            self.stats[y] = sample_weight

    def leaf_prediction(self, x, *, tree=None):
        return normalize_values_in_dict(self.stats, inplace=False)

    def calculate_promise(self):
        raise NotImplementedError

    @staticmethod
    def is_leaf():
        # We enforce this class is treated as a decision node to avoid it is
        # deactivated by the memory management routines.
        return False

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
                count += child.count_nodes()  # noqa

        return count

    def observed_class_distribution_is_pure(self):
        """Check if observed class distribution is pure, i.e. if all samples
        belong to the same class.

        Returns
        -------
            True if the observed number of classes is smaller than 2, False otherwise.
        """
        count = 0
        for weight in self.stats.values():
            if weight != 0:
                count += 1
                if count == 2:  # No need to count beyond this point
                    break
        return count < 2


class EFDTLearningNodeMC(BaseEFDTNode, LearningNodeMC):
    """Active Learning node for the Hoeffding Anytime Tree.

    Parameters
    ----------
    stats
        Initial class observations.
    depth
        The depth of the node.
    splitter
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    kwargs
        Other parameters passed to the learning nodes.
    """

    def __init__(self, stats, depth, splitter, **kwargs):
        super().__init__(stats, depth, splitter, **kwargs)


class EFDTLearningNodeNB(BaseEFDTNode, LearningNodeNB):
    """Learning node  for the Hoeffding Anytime Tree that uses Naive Bayes
    models.

    Parameters
    ----------
    stats
        Initial class observations
    depth
        The depth of the node.
    splitter
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    kwargs
        Other parameters passed to the learning nodes.
    """

    def __init__(self, stats, depth, splitter, **kwargs):
        super().__init__(stats, depth, splitter, **kwargs)


class EFDTLearningNodeNBA(BaseEFDTNode, LearningNodeNBA):
    """Learning node for the Hoeffding Anytime Tree that uses Adaptive Naive
    Bayes models.

    Parameters
    ----------
    stats
        Initial class observations.
    depth
        The depth of the node.
    splitter
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    kwargs
        Other parameters passed to the learning nodes.
    """

    def __init__(self, stats, depth, splitter, **kwargs):
        super().__init__(stats, depth, splitter, **kwargs)
