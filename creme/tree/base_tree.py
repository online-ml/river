from abc import ABC, abstractmethod

import math
import copy
from operator import itemgetter

# TODO review later
from skmultiflow.rules.base_rule import Rule

from creme.utils.skmultiflow_utils import calculate_object_size

from ._nodes import Node
from ._nodes import LearningNode
from ._nodes import ActiveLeaf
from ._nodes import InactiveLeaf
from ._nodes import SplitNode
from ._nodes import FoundNode


class DecisionTree(ABC):
    """ Base class for Decision Trees.

    It defines base operations and properties that all the decision trees must inherit or
    implement according to their own particularities.

    Particularly, the base Decision Tree defines the following standards:

    * TODO

    Parameters
    ----------
    max_depth
    The maximum depth a tree can reach. If `None`, the tree will grow indefinitely.
    max_size
        The max size of the tree, in Megabytes (MB).
    memory_estimate_period
        Interval (number of processed instances) between memory consumption checks.
    stop_mem_management
        If True, stop growing as soon as memory limit is hit.
    remove_poor_atts
        If True, disable poor attributes to reduce memory usage.
    no_preprune
        If True, disable pre-pruning.
    """
    def __init__(self, max_depth: int = None, max_size: int = 1000,
                 memory_estimate_period: int = 1000000, stop_mem_management: bool = False,
                 remove_poor_atts: bool = False, no_preprune: bool = False):
        self.max_depth = max_depth if max_depth is not None else float('Inf')
        self.max_size = max_size
        self.memory_estimate_period = memory_estimate_period
        self.stop_mem_management = stop_mem_management
        self.remove_poor_atts = remove_poor_atts
        self.no_preprune = no_preprune

    def __sizeof__(self):
        """ Calculate the size of the tree.

        Returns
        -------
        int
            Size of the tree in bytes.

        """
        return calculate_object_size(self)

    def reset(self):
        """ Reset the Hoeffding Tree to default values."""
        self._tree_root = None
        self._decision_node_cnt = 0
        self._active_leaf_node_cnt = 0
        self._inactive_leaf_node_cnt = 0
        self._inactive_leaf_byte_size_estimate = 0.0
        self._active_leaf_byte_size_estimate = 0.0
        self._byte_size_estimate_overhead_fraction = 1.0
        self._growth_allowed = True
        self._train_weight_seen_by_model = 0.0

        return self

    @staticmethod
    def _hoeffding_bound(range_val, confidence, n):
        r""" Compute the Hoeffding bound, used to decide how many samples are necessary at each
        node.

        Notes
        -----
        The Hoeffding bound is defined as:

        $\epsilon = \sqrt{\frac{R^2\ln(1/\delta))}{2n}}$

        where:

        $\epsilon$: Hoeffding bound.
        $R$: Range of a random variable. For a probability the range is 1, and for an
        information gain the range is log *c*, where *c* is the number of classes.
        $\delta: Confidence. 1 minus the desired probability of choosing the correct
        attribute at any given node.
        $n$: Number of samples.

        Parameters
        ----------
        range_val
            Range value.
        confidence
            Confidence of choosing the correct attribute.
        n
            Number of processed samples.
        """
        return math.sqrt((range_val * range_val * math.log(1. / confidence)) / (2. * n))

    @property
    def model_measurements(self):
        """ Collect metrics corresponding to the current status of the tree.

        Returns
        -------
        string
            A string buffer containing the measurements of the tree.
        """
        measurements = {'Tree size (nodes)': self._decision_node_cnt
                        + self._active_leaf_node_cnt + self._inactive_leaf_node_cnt,
                        'Tree size (leaves)': self._active_leaf_node_cnt
                        + self._inactive_leaf_node_cnt,
                        'Active learning nodes': self._active_leaf_node_cnt,
                        'Tree depth': self._measure_tree_depth(),
                        'Active leaf byte size estimate': self._active_leaf_byte_size_estimate,
                        'Inactive leaf byte size estimate': self._inactive_leaf_byte_size_estimate,
                        'Byte size estimate overhead': self._byte_size_estimate_overhead_fraction
                        }
        return measurements

    def get_model_description(self):
        """ Walk the tree and return its structure in a buffer.

        Returns
        -------
        string
            The description of the model.

        """
        if self._tree_root is not None:
            buffer = ['']
            description = ''
            self._tree_root.describe_subtree(self, buffer, 0)
            for line in range(len(buffer)):
                description += buffer[line]
            return description

    def _new_split_node(self, split_test, target_stats) -> SplitNode:
        """ Create a new split node."""
        return SplitNode(split_test, target_stats)

    @abstractmethod
    def _new_learning_node(self, initial_stats: dict = None, parent: LearningNode = None,
                           is_active: bool = True) -> LearningNode:
        """ Create a new learning node.

        The characteristics of tje learning node depends on the tree algorithm.

        Parameters
        ----------
        initial_stats
            Target statistics inherited from the parent node.
        parent
            Parent node to inherit from.
        is_active
            Define whether or not the new node to be created is an active learning node.

        Returns
        -------
        node
            A new learning node.
        """

    @property
    def depth(self) -> int:
        """ Calculate the depth of the tree.

        Returns
        -------
        int
            Depth of the tree.
        """
        if isinstance(self._tree_root, Node):
            return self._tree_root.subtree_depth()
        return 0

    @property
    def split_criterion(self) -> str:
        """ Return a string with the name of the split criterion being used by the tree. """
        return self._split_criterion

    @split_criterion.setter
    @abstractmethod
    def split_criterion(self, split_criterion):
        """ Define the split criterion to be used by the tree. """

    @property
    def leaf_prediction(self) -> str:
        """ Return the prediction strategy used by the tree at its leaves. """
        return self._leaf_prediction

    @leaf_prediction.setter
    @abstractmethod
    def leaf_prediction(self, leaf_prediction):
        """ Define the prediction strategy used by the tree in its leaves."""

    def _enforce_size_limit(self):
        """ Track the size of the tree and disable/enable nodes if required."""
        byte_size = (self._active_leaf_byte_size_estimate
                     + self._inactive_leaf_node_cnt * self._inactive_leaf_byte_size_estimate) \
            * self._byte_size_estimate_overhead_fraction
        if self._inactive_leaf_node_cnt > 0 or byte_size > self.max_byte_size:
            if self.stop_mem_management:
                self._growth_allowed = False
                return
        learning_nodes = self._find_learning_nodes()
        learning_nodes = learning_nodes.sort(key=lambda n: n.node.calculate_promise())
        max_active = 0
        while max_active < len(learning_nodes):
            max_active += 1
            if (((max_active * self._active_leaf_byte_size_estimate
                    + (len(learning_nodes) - max_active) * self._inactive_leaf_byte_size_estimate)
                    * self._byte_size_estimate_overhead_fraction) > self.max_byte_size):
                max_active -= 1
                break
        cutoff = len(learning_nodes) - max_active
        for i in range(cutoff):
            if isinstance(learning_nodes[i].node, ActiveLeaf):
                self._deactivate_learning_node(learning_nodes[i].node,
                                               learning_nodes[i].parent,
                                               learning_nodes[i].parent_branch)
        for i in range(cutoff, len(learning_nodes)):
            if isinstance(learning_nodes[i].node, InactiveLeaf):
                self._activate_learning_node(learning_nodes[i].node,
                                             learning_nodes[i].parent,
                                             learning_nodes[i].parent_branch)

    def _estimate_model_size(self):
        """ Calculate the size of the model and trigger tracker function
        if the actual model size exceeds the max size in the configuration."""
        learning_nodes = self._find_learning_nodes()
        total_active_size = 0
        total_inactive_size = 0
        for found_node in learning_nodes:
            if not found_node.node.is_leaf():  # Safety check for non-trivial tree structures
                continue
            if isinstance(found_node.node, ActiveLeaf):
                total_active_size += calculate_object_size(found_node.node)
            else:
                total_inactive_size += calculate_object_size(found_node.node)
        if total_active_size > 0:
            self._active_leaf_byte_size_estimate = total_active_size / self._active_leaf_node_cnt
        if total_inactive_size > 0:
            self._inactive_leaf_byte_size_estimate = total_inactive_size \
                / self._inactive_leaf_node_cnt
        actual_model_size = calculate_object_size(self)
        estimated_model_size = (self._active_leaf_node_cnt * self._active_leaf_byte_size_estimate
                                + self._inactive_leaf_node_cnt
                                * self._inactive_leaf_byte_size_estimate)
        self._byte_size_estimate_overhead_fraction = actual_model_size / estimated_model_size
        if actual_model_size > self.max_byte_size:
            self._enforce_size_limit()

    def _deactivate_all_leaves(self):
        """ Deactivate all leaves. """
        learning_nodes = self._find_learning_nodes()
        for cur_node in learning_nodes:
            if isinstance(cur_node, ActiveLeaf):
                self._deactivate_learning_node(cur_node.node,
                                               cur_node.parent,
                                               cur_node.parent_branch)

    def _deactivate_leaf(self, to_deactivate: ActiveLeaf, parent: SplitNode, parent_branch: int):
        """ Deactivate a learning node.

        Parameters
        ----------
        to_deactivate
            The node to deactivate.
        parent
            The node's parent.
        parent_branch
            Parent node's branch index.
        """
        new_leaf = self._new_learning_node(to_deactivate.stats, is_active=False)
        if parent is None:
            self._tree_root = new_leaf
        else:
            parent.set_child(parent_branch, new_leaf)
        self._active_leaf_node_cnt -= 1
        self._inactive_leaf_node_cnt += 1

    def _activate_leaf(self, to_activate: InactiveLeaf, parent: SplitNode, parent_branch: int):
        """ Activate a learning node.

        Parameters
        ----------
        to_activate
            The node to activate.
        parent
            The node's parent.
        parent_branch
            Parent node's branch index.
        """
        new_leaf = self._new_learning_node(to_activate.stats)
        if parent is None:
            self._tree_root = new_leaf
        else:
            parent.set_child(parent_branch, new_leaf)
        self._active_leaf_node_cnt += 1
        self._inactive_leaf_node_cnt -= 1

    def _find_learning_nodes(self):
        """ Find learning nodes in the tree.

        Returns
        -------
        list
            List of learning nodes in the tree.
        """
        found_list = []
        self.__find_learning_nodes(self._tree_root, None, -1, found_list, 0)
        return found_list

    def __find_learning_nodes(self, node, parent, parent_branch, found, depth):
        """ Find learning nodes in the tree from a given node.

        Parameters
        ----------
        node
            The node to start the search.
        parent
            The node's parent.
        parent_branch
            Parent node's branch.
        depth
            The node's depth.

        Returns
        -------
        list
            List of learning nodes.
        """
        if node is not None:
            if isinstance(node, LearningNode):
                found.append(FoundNode(node, parent, parent_branch, depth))
            if isinstance(node, SplitNode):
                split_node = node
                for i in range(split_node.n_children):
                    self.__find_learning_nodes(
                        split_node.get_child(i), split_node, i, found, depth + 1
                    )

    # TODO review
    def get_model_rules(self):
        """ Returns list of rules describing the tree.

        Returns
        -------
        list (Rule)
            list of the rules describing the tree
        """
        root = self._tree_root
        rules = []

        def recurse(node, cur_rule, ht):
            if isinstance(node, SplitNode):
                for i, child in node._children.items():
                    predicate = node.get_predicate(i)
                    r = copy.deepcopy(cur_rule)
                    r.predicate_set.append(predicate)
                    recurse(child, r, ht)
            else:
                cur_rule.observed_class_distribution = node.stats.copy()
                cur_rule.class_idx = max(node.stats.items(), key=itemgetter(1))[0]
                rules.append(cur_rule)

        rule = Rule()
        recurse(root, rule, self)
        return rules

    def get_rules_description(self):
        """ Prints the description of tree using rules."""
        description = ''
        for rule in self.get_model_rules():
            description += str(rule) + '\n'

        return description
