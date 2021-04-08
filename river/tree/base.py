"""

This module defines generic branch and leaf implementations. These should be used in River by each
tree-based model. Using these classes makes the code more DRY. The only exception for not doing so
would be for performance, whereby a tree-based model uses a bespoke implementation.

This module defines a bunch of methods to ease the manipulation and diagnostic of trees. Its
intention is to provide utilies for walking over a tree and visualizing it.

"""
import abc
from typing import Iterable, Union


class Branch(abc.ABC):
    """A generic tree branch."""

    def __init__(self, *children):
        self.children = children

    @abc.abstractmethod
    def next(self, x) -> Union["Branch", "Leaf"]:
        """Move to the next node down the tree."""

    def walk(self, x) -> Iterable[Union["Branch", "Leaf"]]:
        """Iterate over the nodes that lead to the leaf which contains x."""
        node = self
        while isinstance(node, Branch):
            yield node
            node = node.next(x)
        yield node

    def traverse(self, x) -> "Leaf":
        """Return the leaf corresponding to the given input."""
        for node in self.walk(x):
            pass
        return node

    @property
    def n_nodes(self):
        """Number of descendants, including thyself."""
        return 1 + sum(child.n_nodes for child in self.children)

    @property
    def n_branches(self):
        """Number of branches, including thyself."""
        return 1 + sum(child.n_branches for child in self.children)

    @property
    def n_leaves(self):
        """Number of leaves."""
        return sum(child.n_leaves for child in self.children)

    @property
    def height(self):
        """Distance to the deepest descendant."""
        return 1 + max(child.height for child in self.children)

    def iter_dfs(self):
        """Iterate over nodes in depth-first order."""
        yield self
        for child in self.children:
            yield from child.iter_dfs()

    def iter_leaves(self):
        """Iterate over leaves from the left-most one to the right-most one."""
        for child in self.children:
            yield from child.iter_leaves()

    def iter_branches(self):
        """Iterate over branches in depth-first order."""
        yield self
        for child in self.children:
            yield from child.iter_branches()

    def iter_edges(self):
        """Iterate over edges in depth-first order."""
        for child in self.children:
            yield self, child
            yield from child.iter_edges()


class Leaf:
    """A generic tree node."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @property
    def n_nodes(self):
        return 1

    @property
    def n_branches(self):
        return 0

    @property
    def n_leaves(self):
        return 1

    @property
    def height(self):
        return 1

    def iter_dfs(self):
        yield self

    def iter_leaves(self):
        yield self

    def iter_branches(self):
        yield from ()

    def iter_edges(self):
        yield from ()
