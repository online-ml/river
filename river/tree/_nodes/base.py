import collections
import copy
import numbers
import textwrap
import typing
from abc import ABCMeta, abstractmethod

from river import base
from river.stats import Var

from .._attribute_test import InstanceConditionalTest, SplitSuggestion

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

    def __init__(self, stats: typing.Union[dict, Var] = None, depth: int = 0, **kwargs):
        self.stats: typing.Union[dict, Var] = stats if stats is not None else {}
        self.depth = depth
        self.kwargs = kwargs

    @staticmethod
    def is_leaf() -> bool:
        """Indicate if the node is a leaf.

        Returns
        -------
        True if node is a leaf, False otherwise
        """
        return True

    def filter_instance_to_leaf(
        self, x: dict, parent: "Node", parent_branch: int
    ) -> FoundNode:
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

    def path(self, x) -> typing.Iterator["Node"]:
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

    def describe_subtree(self, tree, buffer: typing.List[str], indent: int = 0):
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
                text += f"{repr(self.stats.mean)} | {repr(self.stats)}"
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
    kwargs
        Other parameters passed to the split node.
    """

    def __init__(self, split_test: InstanceConditionalTest, stats, depth, **kwargs):
        super().__init__(stats, depth, **kwargs)
        self.split_test = split_test
        # dict -> branch_id: child_node
        self._children: typing.Dict[int, Node] = {}

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
        if (self.split_test.max_branches() >= 0) and (
            index >= self.split_test.max_branches()
        ):
            raise IndexError
        self._children[index] = node

    def get_child(self, index: int) -> typing.Union[Node, None]:
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
        return self.split_test.branch_for_instance(x)

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

    def describe_subtree(self, tree, buffer: typing.List[str], indent: int = 0):
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
                buffer[0] += self.split_test.describe_condition_for_branch(branch_idx)
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
    splitter
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    kwargs
        Other parameters passed to the learning node.
    """

    def __init__(self, stats, depth, splitter, **kwargs):
        super().__init__(stats, depth, **kwargs)

        self.splitter = splitter

        self.splitters = {}
        self._disabled_attrs = set()
        self._last_split_attempt_at = self.total_weight

    def is_active(self):
        return self.splitters is not None

    def activate(self):
        if not self.is_active():
            self.splitters = {}

    def deactivate(self):
        self.splitters = None

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

    @staticmethod
    @abstractmethod
    def new_nominal_splitter():
        pass

    @abstractmethod
    def update_stats(self, y, sample_weight):
        pass

    def _iter_features(self, x) -> typing.Iterable:
        """Determine how the input instance is looped through when updating the splitters.

        Parameters
        ----------
        x
            The input instance.
        """
        for att_id, att_val in x.items():
            yield att_id, att_val

    def update_splitters(self, x, y, sample_weight, nominal_attributes):
        for att_id, att_val in self._iter_features(x):
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

    def best_split_suggestions(self, criterion, tree) -> typing.List[SplitSuggestion]:
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
            null_split = SplitSuggestion(
                None, [{}], criterion.merit_of_split(pre_split_dist, [pre_split_dist])
            )
            best_suggestions.append(null_split)
        for att_id, splitter in self.splitters.items():
            best_suggestion = splitter.best_evaluated_split_suggestion(
                criterion, pre_split_dist, att_id, tree.binary_split
            )
            if best_suggestion is not None:
                best_suggestions.append(best_suggestion)
        return best_suggestions

    def disable_attribute(self, att_id):
        """Disable an attribute observer.

        Parameters
        ----------
        att_id
            Attribute index.

        """
        if att_id in self.splitters:
            del self.splitters[att_id]
            self._disabled_attrs.add(att_id)

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
            self.update_splitters(x, y, sample_weight, tree.nominal_attributes)

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
