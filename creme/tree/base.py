import collections


class Split(collections.namedtuple('Split', 'on how at')):
    """A data class for storing split details."""

    def __call__(self, x):
        return self.how(x[self.on], self.at)

    def __repr__(self):
        at = self.at
        if isinstance(at, float):
            at = f'{at:.5f}'
        return f'{self.on} {self.how} {at}'


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
        return self.left if self.split(x) else self.right

    def path(self, x):
        yield self
        yield from self.next(x).path(x)

    @property
    def size(self):
        return self.left.size + self.right.size


class Leaf(Node):

    def path(self, x):
        yield self

    @property
    def size(self):
        return 1
