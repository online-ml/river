import abc

from .. import stats


class Node(abc.ABC):
    """A node is an element of a tree."""

    @abc.abstractmethod
    def apply(self, x):
        """Passes an observation through the tree."""


class Branch(Node):
    """A branch of a tree is a node that has two children."""

    def __init__(self, feature, threshold, left, right):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

    def apply(self, x):
        if x[self.feature] > self.threshold:
            return self.right.apply(x)
        return self.left.apply(x)


class Leaf(Node):
    """A leaf of a tree is a node that has no children."""

    def __init__(self, statistic):
        self.statistic = statistic

    def apply(self, x):
        return self.statistic.get()


class MondrianTree:

    def __init__(self):
        self.tree = Leaf()
