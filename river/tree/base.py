"""

This module defines generic branch and leaf implementations. These should be used in River by each
tree-based model. Using these classes makes the code more DRY. The only exception for not doing so
would be for performance, whereby a tree-based model uses a bespoke implementation.

This module defines a bunch of methods to ease the manipulation and diagnostic of trees. Its
intention is to provide utilities for walking over a tree and visualizing it.

"""
import abc
import typing
from collections import defaultdict
from queue import Queue
from xml.etree import ElementTree as ET

import pandas as pd

from river import base

__all__ = ["Branch", "Leaf"]


class Branch(base.Base, abc.ABC):
    """A generic tree branch.

    Parameters
    ----------
    children
        Child branches and/or leaves.

    """

    def __init__(self, *children):
        self.children = children

    @abc.abstractmethod
    def next(self, x) -> typing.Union["Branch", "Leaf"]:
        """Move to the next node down the tree."""

    @abc.abstractmethod
    def most_common_path(self) -> typing.Tuple[int, typing.Union["Leaf", "Branch"]]:
        """Return a tuple with the branch index and the child node related to the most
        traversed path.

        Used in case the split feature is missing from an instance.
        """
        pass

    @property
    @abc.abstractmethod
    def repr_split(self):
        """String representation of the split."""

    def walk(self, x, until_leaf=True) -> typing.Iterable[typing.Union["Branch", "Leaf"]]:
        """Iterate over the nodes of the path induced by x."""
        yield self
        try:
            yield from self.next(x).walk(x, until_leaf)
        except KeyError:
            if until_leaf:
                _, node = self.most_common_path()
                yield node
                yield from node.walk(x, until_leaf)

    def traverse(self, x, until_leaf=True) -> typing.Union["Branch", "Leaf"]:
        """Return the leaf corresponding to the given input."""
        for node in self.walk(x, until_leaf):
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

    def iter_bfs(self):
        """Iterate over nodes in breadth-first order."""

        queue = Queue()

        queue.put(self)

        while not queue.empty():
            node = queue.get()
            yield node
            if isinstance(node, Branch):
                for child in node.children:
                    queue.put(child)

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

    def to_dataframe(self) -> pd.DataFrame:
        """Build a DataFrame containing one record for each node."""
        node_ids: typing.DefaultDict[typing.Hashable, int] = defaultdict(lambda: len(node_ids))  # type: ignore
        nodes = []

        queue: "Queue" = Queue()
        queue.put((self, None, 0))

        while not queue.empty():
            node, parent, depth = queue.get()
            nodes.append(
                {
                    "node": node_ids[id(node)],
                    "parent": node_ids[id(parent)] if parent else pd.NA,
                    "is_leaf": isinstance(node, Leaf),
                    "depth": depth,
                    **{k: v for k, v in node.__dict__.items() if k != "children"},
                }
            )
            try:
                for child in node.children:
                    queue.put((child, node, depth + 1))
            except AttributeError:
                pass

        return pd.DataFrame.from_records(nodes).set_index("node")

    def _repr_html_(self):
        from river.tree import viz

        div = viz.tree_to_html(self)
        return f"<div>{ET.tostring(div, encoding='unicode')}<style scoped>{viz.CSS}</style></div>"


class Leaf(base.Base):
    """A generic tree node.

    Parameters
    ----------
    kwargs
        Each provided keyword argument is stored in the leaf as an attribute.

    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def walk(self, x, until_leaf=True):  # noqa
        yield self

    @property
    @abc.abstractmethod
    def __repr__(self):
        """String representation for visualization purposes."""

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

    def iter_branches(self):  # noqa
        yield from ()

    def iter_edges(self):  # noqa
        yield from ()
