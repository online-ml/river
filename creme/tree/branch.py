import collections


class Split(collections.namedtuple('Split', 'on how at')):
    """A data class for storing split details."""

    def __str__(self):
        return f'{self.on} {self.how} {self.at}'

    def __call__(self, x):
        return self.how(x[self.on], self.at)


class Branch:

    def __init__(self, split, left, right, tree):
        self.split = split
        self.left = left
        self.right = right
        self.tree = tree

    @property
    def size(self):
        return self.left.size + self.right.size

    def get_leaf(self, x):
        if self.split(x):
            return self.left.get_leaf(x)
        return self.right.get_leaf(x)

    def update(self, x, y):
        if self.split(x):
            self.left = self.left.update(x, y)
            return self
        self.right = self.right.update(x, y)
        return self
