from abc import ABCMeta, abstractmethod
import collections
import numbers
import textwrap

from typing import Dict, Iterator, List, Union


from river import base
from river.stats import Var
from river.tree._attribute_test import InstanceConditionalTest  # noqa
from river.tree._attribute_test import AttributeSplitSuggestion  # noqa


# Helper structure to manage nodes
FoundNode = collections.namedtuple("FoundNode", ["node", "parent", "parent_branch"])


class Node(metaclass=ABCMeta):
    """Base class for nodes in a tree.

    Parameters
    ----------
    stats
        Statistics kept by the node.
    depth
        The depth of the node.
    kwargs
        To insure descendants of this class play nice with each other (in case the trees
        follow non-trivial structures).
    """

    def __init__(self, stats: Union[dict, Var] = None, depth: int = 0, **kwargs):
        self._stats: Union[dict, Var] = stats if stats is not None else {}
        self._depth = depth

    @staticmethod
    def is_leaf() -> bool:
        """Indicate if the node is a leaf.

        Returns
        -------
        True if node is a leaf, False otherwise
        """
        return True

    def filter_instance_to_leaf(self, x: dict, parent: "Node", parent_branch: int) -> FoundNode:
        """Traverse down the tree to locate the corresponding leaf for an instance.

        Parameters
        ----------
        x
           Data instances.
        parent
            Parent node.
        parent_branch
            Parent branch index.

        Returns
        -------
            The corresponding leaf.

        """
        return FoundNode(self, parent, parent_branch)

    def path(self, x) -> Iterator["Node"]:
        """
        Yield the nodes that lead to the leaf which contains x.

        Parameters
        ----------
        x
            Instance to sort down the tree.

        Returns
        -------

        """
        yield self

    def iter_edges(self):
        yield None, 0, None, self, None

    @property
    def stats(self) -> Union[dict, Var]:
        """Statistics observed by the node. """
        return self._stats

    @stats.setter
    def stats(self, new_stats: Union[dict, Var]):
        """Set the statistics at the node. """
        self._stats = new_stats if new_stats is not None else {}

    @property
    def depth(self) -> int:
        return self._depth

    @depth.setter
    def depth(self, depth):
        if depth >= 0:
            self._depth = depth

    @property
    @abstractmethod
    def total_weight(self) -> float:
        """Calculate the total weight seen by the node.

        Returns
        -------
        Total weight seen.
        """
        pass

    def subtree_depth(self) -> int:
        """Calculate the depth of the subtree from this node.

        Returns
        -------
        int
            Subtree depth, 0 if the node is a leaf.
        """
        return 0

    def describe_subtree(self, tree, buffer: List[str], indent: int = 0):
        """Walk the tree and write its structure to a buffer string.

        Parameters
        ----------
        tree
            The tree to describe.
        buffer
            The string buffer where the tree's structure will be stored
        indent
            Indentation level (number of white spaces for current node.)
        """
        buffer[0] += textwrap.indent("Leaf = ", " " * indent)

        if isinstance(tree, base.Classifier):
            class_val = max(self.stats, key=self.stats.get)
            buffer[0] += f"Class {class_val} | {self.stats}\n"
        else:
            text = "{"
            # Multi-target regression case
            if isinstance(tree, base.MultiOutputMixin):
                for i, (target_id, var) in enumerate(self.stats.items()):
                    text += f"{target_id}: {self.stats[target_id].mean} | {self.stats[target_id]}"
                    text += ", " if i < len(self.stats) - 1 else ""
            else:  # Single-target regression
                text += f"{self.stats.mean} | {self.stats}"
            text += "}"
            buffer[0] += f"Output {text}\n"  # Regression problems


class SplitNode(Node):
    """Node that splits the data in a tree.

    Parameters
    ----------
    split_test
        Split test.
    stats
        Class observations.
    depth
        The depth of the node.
    """

    def __init__(self, split_test: InstanceConditionalTest, stats, depth, **kwargs):
        super().__init__(stats, depth, **kwargs)
        self._split_test = split_test
        # Dict -> branch_id: child_node
        self._children: Dict[int, Node] = {}

    @staticmethod
    def is_leaf():
        """Determine if the node is a leaf.

        Returns
        -------
        boolean
            True if node is a leaf, False otherwise
        """
        return False

    @property
    def n_children(self) -> int:
        """Count the number of children for a node."""
        return len(self._children)

    @property
    def split_test(self) -> InstanceConditionalTest:
        """The split test of this node.

        Returns
        -------
        InstanceConditionalTest
            Split test.
        """

        return self._split_test

    @property
    def total_weight(self) -> float:
        """Calculate the total weight seen by the node.

        Returns
        -------
        Total weight seen.
        """
        return sum(ch.total_weight for ch in filter(None, self._children.values()))

    def set_child(self, index: int, node: Node):
        """Set node as child.

        Parameters
        ----------
        index
            Branch index where the node will be inserted.
        node
            The node to insert.
        """
        if (self._split_test.max_branches() >= 0) and (index >= self._split_test.max_branches()):
            raise IndexError
        self._children[index] = node

    def get_child(self, index: int) -> Union[Node, None]:
        """Retrieve a node's child given its branch index.

        Parameters
        ----------
        index
            Node's branch index.

        Returns
        -------
            Child node or None if the corresponding branch index does not exists.
        """
        if index in self._children:
            return self._children[index]
        else:
            return None

    def instance_child_index(self, x: dict) -> int:
        """Get the branch index for a given instance at the current node.

        Returns
        -------
        int
            Branch index, -1 if unknown.
        """
        return self._split_test.branch_for_instance(x)

    def filter_instance_to_leaf(self, x: dict, parent: Node, parent_branch: int):
        """Traverse down the tree to locate the corresponding leaf for an instance.

        Parameters
        ----------
        x
           Data instance.
        parent
            Parent node.
        parent_branch
            Parent branch index.

        Returns
        -------
            Leaf node for the instance.

        """
        child_index = self.instance_child_index(x)
        if child_index >= 0:
            child = self.get_child(child_index)
            if child is not None:
                return child.filter_instance_to_leaf(x, self, child_index)
            else:
                return FoundNode(None, self, child_index)
        else:
            return FoundNode(self, parent, parent_branch)

    def path(self, x):
        yield self
        child_index = self.instance_child_index(x)
        if child_index >= 0:
            child = self.get_child(child_index)
            if child is not None:
                yield from child.path(x)

    def iter_edges(self):
        """Iterate over edges in a depth-first manner.

        Returns
        -------
            Tuples in the form (parent_no, child_id, parent, child, branch_id).
        """
        counter = 0

        def iterate(node):

            nonlocal counter
            no = counter

            if not node.is_leaf():

                for branch_id, child in node._children.items():  # noqa
                    counter += 1
                    yield no, counter, node, child, branch_id
                    if not child.is_leaf():
                        yield from iterate(child)

        yield None, 0, None, self, None
        yield from iterate(self)

    def subtree_depth(self) -> int:
        """Calculate the depth of the subtree from this node.

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

    def describe_subtree(self, tree, buffer: List[str], indent: int = 0):
        """Walk the tree and write its structure to a buffer string.

        Parameters
        ----------
        tree
            The tree to describe.
        buffer
            The buffer where the tree's structure will be stored.
        indent
            Indentation level (number of white spaces for current node).
        """
        for branch_idx in range(self.n_children):
            child = self.get_child(branch_idx)
            if child is not None:
                buffer[0] += textwrap.indent("if ", " " * indent)
                buffer[0] += self._split_test.describe_condition_for_branch(branch_idx)
                buffer[0] += ":\n"
                child.describe_subtree(tree, buffer, indent + 2)


class LearningNode(Node, metaclass=ABCMeta):
    """Base Learning Node to be used in Hoeffding Trees.

    Parameters
    ----------
    stats
        Target statistics (they differ in classification and regression tasks).
    depth
        The depth of the node
    attr_obs
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    attr_obs_params
        The parameters passed to the numeric attribute observer algorithm.
    """

    def __init__(self, stats, depth, attr_obs: str, attr_obs_params: dict, **kwargs):
        super().__init__(stats, depth, **kwargs)

        self.attr_obs = attr_obs
        self.attr_obs_params = attr_obs_params

        self._attribute_observers = {}
        self._disabled_attrs = set()
        self._last_split_attempt_at = self.total_weight

    def is_active(self):
        return self._attribute_observers is not None

    def activate(self):
        if not self.is_active():
            self._attribute_observers = {}

    def deactivate(self):
        self._attribute_observers = None

    @property
    def last_split_attempt_at(self) -> float:
        """The weight seen at last split evaluation.

        Returns
        -------
        Weight seen at last split evaluation.
        """
        return self._last_split_attempt_at

    @last_split_attempt_at.setter
    def last_split_attempt_at(self, weight):
        """Set the weight seen at last split evaluation.

        Parameters
        ----------
        weight
            Weight seen at last split evaluation.
        """
        self._last_split_attempt_at = weight

    @property
    def attribute_observers(self):
        return self._attribute_observers

    @attribute_observers.setter
    def attribute_observers(self, attr_obs):
        self._attribute_observers = attr_obs

    @staticmethod
    @abstractmethod
    def new_nominal_attribute_observer():
        pass

    @staticmethod
    @abstractmethod
    def new_numeric_attribute_observer(attr_obs, attr_obs_params):
        pass

    @abstractmethod
    def update_stats(self, y, sample_weight):
        pass

    def update_attribute_observers(self, x, y, sample_weight, nominal_attributes):
        for attr_idx, attr_val in x.items():
            if attr_idx in self._disabled_attrs:
                continue

            try:
                obs = self.attribute_observers[attr_idx]
            except KeyError:
                if (
                    nominal_attributes is not None and attr_idx in nominal_attributes
                ) or not isinstance(attr_val, numbers.Number):
                    obs = self.new_nominal_attribute_observer()
                else:
                    obs = self.new_numeric_attribute_observer(
                        attr_obs=self.attr_obs, attr_obs_params=self.attr_obs_params
                    )
                self.attribute_observers[attr_idx] = obs
            obs.update(attr_val, y, sample_weight)

    def best_split_suggestions(self, criterion, tree) -> List[AttributeSplitSuggestion]:
        """Find possible split candidates.

        Parameters
        ----------
        criterion
            The splitting criterion to be used.
        tree
            Decision tree.

        Returns
        -------
        Split candidates.
        """
        best_suggestions = []
        pre_split_dist = self.stats
        if tree.merit_preprune:
            # Add null split as an option
            null_split = AttributeSplitSuggestion(
                None, [{}], criterion.merit_of_split(pre_split_dist, [pre_split_dist])
            )
            best_suggestions.append(null_split)
        for i, obs in self.attribute_observers.items():
            best_suggestion = obs.best_evaluated_split_suggestion(
                criterion, pre_split_dist, i, tree.binary_split
            )
            if best_suggestion is not None:
                best_suggestions.append(best_suggestion)
        return best_suggestions

    def disable_attribute(self, attr_idx):
        """Disable an attribute observer.

        Parameters
        ----------
        attr_idx
            Attribute index.

        """
        if attr_idx in self.attribute_observers:
            del self.attribute_observers[attr_idx]
            self._disabled_attrs.add(attr_idx)

    def learn_one(self, x, y, *, sample_weight=1.0, tree=None):
        """Update the node with the provided sample.

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

        Notes
        -----
        This base implementation defines the basic functioning of a learning node.
        All classes overriding this method should include a call to `super().learn_one`
        to guarantee the learning process happens consistently.
        """
        self.update_stats(y, sample_weight)
        if self.is_active():
            self.update_attribute_observers(x, y, sample_weight, tree.nominal_attributes)

    @abstractmethod
    def leaf_prediction(self, x, *, tree=None) -> dict:
        pass

    @abstractmethod
    def calculate_promise(self) -> int:
        """Calculate node's promise.

        Returns
        -------
        int
            A small value indicates that the node has seen more samples of a
            given class than the other classes.

        """
