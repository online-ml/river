"""Generic branch and leaf implementation."""
import collections


class Split(collections.namedtuple('Split', 'on how at')):
    """A data class for storing split details."""

    def __call__(self, x):
        return self.how(x[self.on], self.at)

    def __repr__(self):
        at = self.at
        if isinstance(at, float):
            at = f'{at:.5f}'
        return f'{self.on} {self.how.__name__} {at}'


class Node:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Branch(Node):

    def __init__(self, split, left, right, **kwargs):
        super().__init__(**kwargs)
        self.split = split
        self.left = left
        self.right = right

    def next(self, x):
        """Returns the next node where x belongs."""
        return self.left if self.split(x) else self.right

    def path(self, x):
        """Iterates over the nodes that lead to the leaf which contains x."""
        yield self
        yield from self.next(x).path(x)

    @property
    def size(self):
        """Number of descendants, including thyself."""
        return 1 + self.left.size + self.right.size

    @property
    def height(self):
        """Distance to the deepest node."""
        return 1 + max(self.left.height, self.right.height)

    def _iter_dfs(self, depth=0):
        """Iterates over nodes via depth-first search.

        Example:

            >>> tree = Branch(
            ...     None,
            ...     Branch(
            ...         None,
            ...         Leaf(no=2),
            ...         Leaf(no=3),
            ...         no=1
            ...     ),
            ...     Leaf(no=4),
            ...     no=0
            ... )

            >>> for node, depth in tree._iter_dfs():
            ...     print(f'#{node.no}, depth {depth}')
            #0, depth 0
            #1, depth 1
            #2, depth 2
            #3, depth 2
            #4, depth 1

        """
        yield self, depth
        yield from self.left._iter_dfs(depth=depth + 1)
        yield from self.right._iter_dfs(depth=depth + 1)


class Leaf(Node):

    def path(self, x):
        yield self

    @property
    def size(self):
        return 1

    @property
    def height(self):
        return 1

    def _iter_dfs(self, depth):
        yield self, depth


def iter_blocks(tree, limits, depth=-1):
    """Returns the block which encloses each node at a given depth.

    This only makes sense if the branches of the provided tree use the ``<`` operator as a split
    rule.

    Parameters:
        tree (Node)
        limits (dict)
        depth (int): Desired tree depth. Set to ``-1`` to iterate over the leaves.

    Example:

        >>> import operator

        >>> tree = Branch(
        ...     Split('x', operator.lt, .5),
        ...     Leaf(no=0),
        ...     Branch(
        ...         Split('y', operator.lt, .5),
        ...         Leaf(no=1),
        ...         Leaf(no=2)
        ...     )
        ... )

        >>> for leaf, block in iter_blocks(tree, limits={'x': (0, 1), 'y': (0, 1)}):
        ...     print(leaf.no, block)
        0 {'x': (0, 0.5), 'y': (0, 1)}
        1 {'x': (0.5, 1), 'y': (0, 0.5)}
        2 {'x': (0.5, 1), 'y': (0.5, 1)}

    """

    if depth == 0 or isinstance(tree, Leaf):
        yield (tree, limits)
    else:
        on, at = tree.split.on, tree.split.at

        l_limits = {**limits, on: (limits[on][0], at)} if on in limits else limits
        yield from iter_blocks(tree=tree.left, limits=l_limits, depth=depth - 1)

        r_limits = {**limits, on: (at, limits[on][1])} if on in limits else limits
        yield from iter_blocks(tree=tree.right, limits=r_limits, depth=depth - 1)
