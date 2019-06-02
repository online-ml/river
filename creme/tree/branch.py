class Branch:

    __slots__ = 'split', 'left', 'right', 'tree'

    def __init__(self, split, left, right, tree):
        self.split = split
        self.left = left
        self.right = right
        self.tree = tree

    def get_leaf(self, x):
        if self.split.test(x):
            return self.left.get_leaf(x)
        return self.right.get_leaf(x)

    def update(self, x, y):
        if self.split.test(x):
            self.left = self.left.update(x, y)
            return self
        self.right = self.right.update(x, y)
        return self
