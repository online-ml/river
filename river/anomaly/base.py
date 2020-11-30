"""Generic branch and leaf implementation."""
import operator
import textwrap
import typing


class Op:
    """An operator that is part of a split."""

    __slots__ = "symbol", "func"

    def __init__(self, symbol, func):
        self.symbol = symbol
        self.func = func

    def __call__(self, a, b):
        return self.func(a, b)

    def __repr__(self):
        return self.symbol


LT = Op("<", operator.lt)
EQ = Op("=", operator.eq)


class Split:
    """A split that is part of a branch."""

    def __init__(self, on, how, at):
        self.on = on
        self.how = how
        self.at = at

    def __call__(self, x):
        return self.how(x[self.on], self.at)

    def __repr__(self):
        at = self.at
        if isinstance(at, float):
            at = f"{at:.5f}"
        return f"{self.on} {repr(self.how)} {at}"


class Node:
    """A node is a branch or a leaf."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Branch(Node):
    """A branch is made up of a split and two child nodes."""

    def __init__(self, split, left, right, **kwargs):
        super().__init__(**kwargs)
        self.split = split
        self.left = left
        self.right = right

    def next(self, x):
        """Returns the next node where x belongs."""
        return self.left if self.split(x) else self.right

    def path(self, x):
        """Iterate over the nodes that lead to the leaf which contains x."""
        yield self
        yield from self.next(x).path(x)

    def __repr__(self):
        """

        Examples
        --------

        >>> tree = Branch(
        ...     Split('x', LT, 17.42),
        ...     Branch(
        ...         Split('y', LT, -4.38),
        ...         Leaf(no=2),
        ...         Leaf(no=3),
        ...         no=1
        ...     ),
        ...     Leaf(no=4),
        ...     no=0
        ... )

        >>> tree
        x < 17.42000 {'no': 0}
          y < -4.38000 {'no': 1}
            {'no': 2}
            {'no': 3}
          {'no': 4}

        """
        return (
            repr(self.split)
            + " "
            + str({k: v for k, v in self.__dict__.items() if k not in ("split", "left", "right")})
            + textwrap.indent(f"\n{self.left}\n{self.right}", prefix=" " * 2)
        )

    @property
    def size(self):
        """Number of descendants, including thyself."""
        return 1 + self.left.size + self.right.size

    @property
    def height(self):
        """Distance to the deepest node."""
        return 1 + max(self.left.height, self.right.height)

    def iter_dfs(self, depth=0):
        """Iterate over nodes in a depth-first manner.

        Examples
        --------

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

        >>> for node, depth in tree.iter_dfs():
        ...     print(f'#{node.no}, depth {depth}')
        #0, depth 0
        #1, depth 1
        #2, depth 2
        #3, depth 2
        #4, depth 1

        """
        yield self, depth
        yield from self.left.iter_dfs(depth=depth + 1)
        yield from self.right.iter_dfs(depth=depth + 1)

    def iter_leaves(self):
        """Iterate over the leaves in a depth-first manner.

        Examples
        --------

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

        >>> for leaf in tree.iter_leaves():
        ...     print(f'#{leaf.no}')
        #2
        #3
        #4

        """
        yield from self.left.iter_leaves()
        yield from self.right.iter_leaves()

    def iter_branches(self):
        """Iterate over branches in a depth-first manner.

        Examples
        --------

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

        >>> for branch in tree.iter_branches():
        ...     print(f'#{branch.no}')
        #0
        #1

        """
        yield self
        yield from self.left.iter_branches()
        yield from self.right.iter_branches()

    def iter_edges(self):
        """Iterate over edges in a depth-first manner.

        Examples
        --------

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

        >>> for parent_no, child_no, parent, child, child_depth in tree.iter_edges():
        ...     print(parent_no, child_no, child_depth)
        None 0 0
        0 1 1
        1 2 2
        1 3 2
        0 4 1

        """
        counter = 0

        def iterate(node, depth):

            nonlocal counter
            no = counter

            if isinstance(node, Branch):
                for child in (node.left, node.right):
                    counter += 1
                    yield no, counter, node, child, depth + 1
                    if isinstance(child, Branch):
                        yield from iterate(child, depth=depth + 1)

        yield None, 0, None, self, 0
        yield from iterate(self, depth=0)

    def iter_blocks(
        self, limits: typing.Dict[typing.Hashable, typing.Tuple[float, float]], depth=-1
    ):
        """Iterate over the blocks which enclose each node.

        This only makes sense if the branches of the provided tree use the `<` operator as a
        split rule.

        Parameters
        ----------
        limits
        depth
            Desired tree depth. Set to `-1` to iterate over the leaves.

        Examples
        --------

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

        >>> for leaf, block in tree.iter_blocks(limits={'x': (0, 1), 'y': (0, 1)}):
        ...     print(leaf.no, block)
        0 {'x': (0, 0.5), 'y': (0, 1)}
        1 {'x': (0.5, 1), 'y': (0, 0.5)}
        2 {'x': (0.5, 1), 'y': (0.5, 1)}

        """
        if depth == 0:
            yield (self, limits)
            return

        on, at = self.split.on, self.split.at

        l_limits = {**limits, on: (limits[on][0], at)} if on in limits else limits
        yield from self.left.iter_blocks(limits=l_limits, depth=depth - 1)

        r_limits = {**limits, on: (at, limits[on][1])} if on in limits else limits
        yield from self.right.iter_blocks(limits=r_limits, depth=depth - 1)

    def iter_splits(self, limits: typing.Dict[typing.Hashable, typing.Tuple[float, float]]):
        """Iterate over splits.

        This only makes sense if the branches of the provided tree use the `<` operator as a split
        rule.

        Parameters
        ----------
        limits
        depth

        Examples
        --------

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

        >>> for line in tree.iter_splits({'x': (0, 1), 'y': (0, 1)}):
        ...     print(line)
        {'x': (0.5, 0.5), 'y': (0, 1)}
        {'x': (0.5, 1), 'y': (0.5, 0.5)}

        """

        on, at = self.split.on, self.split.at

        yield {**limits, on: (at, at)}

        yield from self.left.iter_splits(limits={**limits, on: (limits[on][0], at)})
        yield from self.right.iter_splits(limits={**limits, on: (at, limits[on][1])})


class Leaf(Node):
    def __repr__(self):
        return str(self.__dict__)

    def path(self, x):
        yield self

    @property
    def size(self):
        return 1

    @property
    def height(self):
        return 0

    def iter_dfs(self, depth=0):
        yield self, depth

    def iter_leaves(self):
        yield self

    def iter_branches(self):
        yield from ()

    def iter_edges(self):
        yield None, 0, None, self, 0

    def iter_blocks(self, limits, depth=-1):
        yield self, limits

    def iter_splits(self, limits):
        yield from ()
