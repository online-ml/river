import copy
import numbers

from river.utils.skmultiflow_utils import normalize_values_in_dict

from ..splitter.nominal_splitter_classif import NominalSplitterClassif
from .branch import (
    HTBranch,
    NominalBinaryBranch,
    NominalMultiwayBranch,
    NumericBinaryBranch,
    NumericMultiwayBranch,
)
from .htc_nodes import LeafMajorityClass, LeafNaiveBayes, LeafNaiveBayesAdaptive
from .leaf import HTLeaf


class BaseEFDTLeaf(HTLeaf):
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
        Other parameters passed to the node.
    """

    def __init__(self, stats, depth, splitter, **kwargs):
        super().__init__(stats=stats, depth=depth, splitter=splitter, **kwargs)

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
            best_suggestions.append(best_suggestion)

        return best_suggestions


class BaseEFDTBranch(HTBranch):
    """Node that splits the data in a EFDT.

    This node is an exception among the tree's nodes. EFDTSplitNode is both a split node
    and a learning node. EFDT updates all of the nodes in the path from the root to a leaf
    when a new instance arrives. Besides that, it also revisit split decisions from time
    to time. For that reason, this decision node also needs to be able to learn from new
    instances.

    Parameters
    ----------
    stats
        Class observations
    children
        The children nodes.
    attributes
        Other parameters passed to the learning nodes.
    """

    def __init__(self, stats, *children, splitter, splitters, **attributes):
        super().__init__(stats, *children, **attributes)

        self.splitter = splitter
        self.splitters = splitters

        self._disabled_attrs = set()
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

    def update_splitters(self, x, y, sample_weight, nominal_attributes):
        for att_id, att_val in x.items():
            if att_id in self._disabled_attrs:
                continue

            try:
                splitter = self.splitters[att_id]
            except KeyError:
                if (
                    nominal_attributes is not None and att_id in nominal_attributes
                ) or not isinstance(att_val, numbers.Number):
                    splitter = self.new_nominal_splitter()
                else:
                    splitter = copy.deepcopy(self.splitter)

                self.splitters[att_id] = splitter
            splitter.update(att_val, y, sample_weight)

    def learn_one(self, x, y, *, sample_weight=1.0, tree=None):
        """Update branch with the provided sample.

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
        self.update_splitters(x, y, sample_weight, tree.nominal_attributes)

    def prediction(self, x, *, tree=None):
        return normalize_values_in_dict(self.stats, inplace=False)

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
        x_current = None
        for suggestion in split_suggestions:
            if suggestion.feature == id_att:
                x_current = suggestion
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


class EFDTLeafMajorityClass(BaseEFDTLeaf, LeafMajorityClass):
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


class EFDTLeafNaiveBayes(BaseEFDTLeaf, LeafNaiveBayes):
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


class EFDTLeafNaiveBayesAdaptive(BaseEFDTLeaf, LeafNaiveBayesAdaptive):
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


class EFDTNominalBinaryBranch(BaseEFDTBranch, NominalBinaryBranch):
    def __init__(self, stats, feature, value, depth, left, right, **attributes):
        super().__init__(stats, feature, value, depth, left, right, **attributes)


class EFDTNominalMultiwayBranch(BaseEFDTBranch, NominalMultiwayBranch):
    def __init__(self, stats, feature, feature_values, depth, *children, **attributes):
        super().__init__(stats, feature, feature_values, depth, *children, **attributes)


class EFDTNumericBinaryBranch(BaseEFDTBranch, NumericBinaryBranch):
    def __init__(self, stats, feature, threshold, depth, left, right, **attributes):
        super().__init__(stats, feature, threshold, depth, left, right, **attributes)


class EFDTNumericMultiwayBranch(BaseEFDTBranch, NumericMultiwayBranch):
    def __init__(
        self, stats, feature, radius_and_slots, depth, *children, **attributes
    ):
        super().__init__(
            stats, feature, radius_and_slots, depth, *children, **attributes
        )
