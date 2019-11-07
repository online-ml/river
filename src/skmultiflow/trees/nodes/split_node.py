import textwrap
from skmultiflow.trees.nodes import Node
from skmultiflow.trees.nodes import FoundNode


class SplitNode(Node):
    """ Node that splits the data in a Hoeffding Tree.

    Parameters
    ----------
    split_test: InstanceConditionalTest
        Split test.
    class_observations: dict (class_value, weight) or None
        Class observations

    """

    def __init__(self, split_test, class_observations):
        """ SplitNode class constructor."""
        super().__init__(class_observations)
        self._split_test = split_test
        # Dict of tuples (branch, child)
        self._children = {}

    def num_children(self):
        """ Count the number of children for a node."""
        return len(self._children)

    def get_split_test(self):
        """ Retrieve the split test of this node.

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
        for child in self._children:
            if child is not None:
                depth = child.subtree_depth()
                if depth > max_child_depth:
                    max_child_depth = depth
        return max_child_depth + 1

    def describe_subtree(self, ht, buffer, indent=0):
        """ Walk the tree and write its structure to a buffer string.

        Parameters
        ----------
        ht: HoeffdingTree
            The tree to describe.
        buffer: string
            The buffer where the tree's structure will be stored.
        indent: int
            Indentation level (number of white spaces for current node).

        """
        for branch_idx in range(self.num_children()):
            child = self.get_child(branch_idx)
            if child is not None:
                buffer[0] += textwrap.indent('if ', ' ' * indent)
                buffer[0] += self._split_test.describe_condition_for_branch(branch_idx)
                buffer[0] += ':\n'
                child.describe_subtree(ht, buffer, indent + 2)

    def get_predicate(self, branch):

        return self._split_test.branch_rule(branch)
