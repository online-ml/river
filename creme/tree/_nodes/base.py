from abc import ABCMeta, abstractmethod
import textwrap

from skmultiflow.trees._attribute_test import AttributeSplitSuggestion
from skmultiflow.trees._attribute_observer import AttributeObserverNull


class FoundNode(object):
    """ Base class for tree nodes.

    Parameters
    ----------
    node: SplitNode or LearningNode
        The node object.
    parent: SplitNode or None
        The node's parent.
    parent_branch: int
        The parent node's branch.
    depth: int
        Depth of the tree where the node is located.

    """

    def __init__(self, node=None, parent=None, parent_branch=None, depth=None):
        self.node = node
        self.parent = parent
        self.parent_branch = parent_branch
        self.depth = depth


class Node(metaclass=ABCMeta):
    """ Base class for nodes in a tree.

    Parameters
    ----------
    stats: dict or None
        Statistics kept by the node.

    """

    def __init__(self, stats=None):
        self.stats = stats

    @staticmethod
    def is_leaf():
        """ Determine if the node is a leaf.

        Returns
        -------
        bool
            True if leaf, False otherwise

        """
        return True

    def filter_instance_to_leaf(self, X, parent, parent_branch):
        """ Traverse down the tree to locate the corresponding leaf for an instance.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
           Data instances.
        parent: skmultiflow.trees.nodes.Node or None
            Parent node.
        parent_branch: Int
            Parent branch index

        Returns
        -------
        FoundNode
            The corresponding leaf.

        """
        return FoundNode(self, parent, parent_branch)

    @property
    def stats(self):
        """ Statistics observed by the node.
        """
        return self._stats

    @stats.setter
    def stats(self, stats):
        """ Set the statistics at the node.

        """
        self._stats = stats if stats is not None else {}

    def get_class_votes(self, X, tree):
        """ Get the votes per class for a given instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
           Data instances.
        tree: HoeffdingTreeClassifier
            The tree object.

        Returns
        -------
        dict
            Class votes for the given instance.

        """
        return self._stats

    def subtree_depth(self):
        """ Calculate the depth of the subtree from this node.

        Returns
        -------
        int
            Subtree depth, 0 if the node is a leaf.

        """
        return 0

    def calculate_promise(self):
        """ Calculate node's promise.

        Returns
        -------
        int
            A small value indicates that the node has seen more samples of a
            given class than the other classes.

        """
        total_seen = sum(self._stats.values())
        if total_seen > 0:
            return total_seen - max(self._stats.values())
        else:
            return 0

    def describe_subtree(self, tree, buffer, indent=0):
        """ Walk the tree and write its structure to a buffer string.

        Parameters
        ----------
        tree: HoeffdingTreeClassifier
            The tree to describe.
        buffer: string
            The string buffer where the tree's structure will be stored
        indent: int
            Indentation level (number of white spaces for current node.)

        """
        buffer[0] += textwrap.indent('Leaf = ', ' ' * indent)

        if tree._estimator_type == 'classifier':
            class_val = max(
                self._stats,
                key=self._stats.get
            )
            buffer[0] += 'Class {} | {}\n'.format(
                class_val, self._stats
            )
        else:
            text = '{'
            for i, (k, v) in enumerate(self._stats.items()):
                # Multi-target regression case
                if hasattr(v, 'shape') and len(v.shape) > 0:
                    text += '{}: ['.format(k)
                    text += ', '.join(['{:.4f}'.format(e) for e in v.tolist()])
                    text += ']'
                else:  # Single-target regression
                    text += '{}: {:.4f}'.format(k, v)
                text += ', ' if i < len(self._stats) - 1 else ''
            text += '}'
            buffer[0] += 'Statistics {}\n'.format(text)  # Regression problems


class SplitNode(Node):
    """ Node that splits the data in a tree.

    Parameters
    ----------
    split_test: InstanceConditionalTest
        Split test.
    stats: dict (class_value, weight) or None
        Class observations

    """

    def __init__(self, split_test, stats):
        super().__init__(stats)
        self._split_test = split_test
        # Dict of tuples (branch, child)
        self._children = {}

    @property
    def n_children(self):
        """ Count the number of children for a node."""
        return len(self._children)

    @property
    def split_test(self):
        """ The split test of this node.

        Returns
        -------
        InstanceConditionalTest
            Split test.

        """

        return self._split_test

    def set_child(self, index, node):
        """ Set node as child.

        Parameters
        ----------
        index: int
            Branch index where the node will be inserted.

        node: skmultiflow.trees.nodes.Node
            The node to insert.

        """
        if (self._split_test.max_branches() >= 0) and (index >= self._split_test.max_branches()):
            raise IndexError
        self._children[index] = node

    def get_child(self, index):
        """ Retrieve a node's child given its branch index.

        Parameters
        ----------
        index: int
            Node's branch index.

        Returns
        -------
        skmultiflow.trees.nodes.Node or None
            Child node.

        """
        if index in self._children:
            return self._children[index]
        else:
            return None

    def instance_child_index(self, X):
        """ Get the branch index for a given instance at the current node.

        Returns
        -------
        int
            Branch index, -1 if unknown.

        """
        return self._split_test.branch_for_instance(X)

    @staticmethod
    def is_leaf():
        """ Determine if the node is a leaf.

        Returns
        -------
        boolean
            True if node is a leaf, False otherwise

        """
        return False

    def filter_instance_to_leaf(self, X, parent, parent_branch):
        """ Traverse down the tree to locate the corresponding leaf for an instance.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
           Data instances.
        parent: skmultiflow.trees.nodes.Node
            Parent node.
        parent_branch: int
            Parent branch index.

        Returns
        -------
        FoundNode
            Leaf node for the instance.

        """
        child_index = self.instance_child_index(X)
        if child_index >= 0:
            child = self.get_child(child_index)
            if child is not None:
                return child.filter_instance_to_leaf(X, self, child_index)
            else:
                return FoundNode(None, self, child_index)
        else:
            return FoundNode(self, parent, parent_branch)

    def subtree_depth(self):
        """ Calculate the depth of the subtree from this node.

        Returns
        -------
        int
            Subtree depth, 0 if node is a leaf.
        """
        max_child_depth = 0
        for child in self._children.values():
            if child is not None:
                depth = child.subtree_depth()
                if depth > max_child_depth:
                    max_child_depth = depth
        return max_child_depth + 1

    def describe_subtree(self, tree, buffer, indent=0):
        """ Walk the tree and write its structure to a buffer string.

        Parameters
        ----------
        tree: HoeffdingTreeClassifier
            The tree to describe.
        buffer: string
            The buffer where the tree's structure will be stored.
        indent: int
            Indentation level (number of white spaces for current node).

        """
        for branch_idx in range(self.n_children):
            child = self.get_child(branch_idx)
            if child is not None:
                buffer[0] += textwrap.indent('if ', ' ' * indent)
                buffer[0] += self._split_test.describe_condition_for_branch(branch_idx)
                buffer[0] += ':\n'
                child.describe_subtree(tree, buffer, indent + 2)

    def get_predicate(self, branch):
        return self._split_test.branch_rule(branch)


class LearningNode(Node, metaclass=ABCMeta):
    """ Base Learning Node to be used in Hoeffding Trees.

    Parameters
    ----------
    initial_stats: dict (class_value, weight) or None
        Initial stats (they differ in classification and regression tasks).
    """
    def __init__(self, initial_stats):
        super().__init__(initial_stats)
        self.last_split_attempt_at = self.total_weight

    @abstractmethod
    def update_stats(self, y, weight):
        pass

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
        self.update_stats(y, weight)
        self.update_attribute_observers(X, y, weight, tree)

    @abstractmethod
    def predict_one(self, X, *, tree=None):
        pass

    @property
    @abstractmethod
    def total_weight(self):
        """ Calculate the total weight seen by the node.

        Returns
        -------
        float
            Total weight seen.

        """
        pass

    @property
    def last_split_attempt_at(self):
        """ The weight seen at last split evaluation.

        Returns
        -------
        float
            Weight seen at last split evaluation.

        """
        try:
            return self._last_split_attempt_at
        except AttributeError:
            self._last_split_attempt_at = None      # noqa
            return self._last_split_attempt_at

    @last_split_attempt_at.setter
    def last_split_attempt_at(self, weight):
        """ Set the weight seen at last split evaluation.

        Parameters
        ----------
        weight: float
            Weight seen at last split evaluation.

        """
        self._last_split_attempt_at = weight


class ActiveLeaf(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def new_nominal_attribute_observer():
        pass

    @staticmethod
    @abstractmethod
    def new_numeric_attribute_observer():
        pass

    @property
    def attribute_observers(self):
        try:
            return self._attribute_observers
        except AttributeError:
            self._attribute_observers = {}         # noqa
            return self._attribute_observers

    @attribute_observers.setter
    def attribute_observers(self, attr_obs):
        self._attribute_observers = attr_obs       # noqa

    def update_attribute_observers(self, X, y, weight, tree):
        for idx, x in enumerate(X):
            try:
                obs = self.attribute_observers[idx]
            except KeyError:
                if tree.nominal_attributes is not None and idx in tree.nominal_attributes:
                    obs = self.new_nominal_attribute_observer()
                else:
                    obs = self.new_numeric_attribute_observer()
                self.attribute_observers[idx] = obs
            obs.update(x, y, weight)

    def get_best_split_suggestions(self, criterion, tree):
        """ Find possible split candidates.

        Parameters
        ----------
        criterion: SplitCriterion
            The splitting criterion to be used.
        tree:
            Hoeffding Tree.

        Returns
        -------
        list
            Split candidates.

        """
        best_suggestions = []
        pre_split_dist = self._stats
        if not tree.no_preprune:
            # Add null split as an option
            null_split = AttributeSplitSuggestion(
                None, [{}], criterion.get_merit_of_split(pre_split_dist, [pre_split_dist])
            )
            best_suggestions.append(null_split)
        for i, obs in self.attribute_observers.items():
            best_suggestion = obs.get_best_evaluated_split_suggestion(
                criterion, pre_split_dist, i, tree.binary_split
            )
            if best_suggestion is not None:
                best_suggestions.append(best_suggestion)
        return best_suggestions

    def disable_attribute(self, att_idx):
        """ Disable an attribute observer.

        Parameters
        ----------
        att_idx: int
            Attribute index.

        """
        if att_idx in self.attribute_observers:
            self.attribute_observers[att_idx] = AttributeObserverNull()


class InactiveLeaf:
    @staticmethod
    def new_nominal_attribute_observer():
        return None

    @staticmethod
    def new_numeric_attribute_observer():
        return None

    def update_attribute_observers(self, X, y, weight, tree):
        # An inactive learning nodes does nothing here
        # We use it as a dummy class
        pass
