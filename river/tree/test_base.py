from river.tree.base import Branch, Leaf


class BinaryBranch(Branch):

    def __init__(self, left, right, threshold=None, **kwargs):
        super().__init__(left, right)
        self.threshold = threshold
        self.__dict__.update(kwargs)

    def next(self, x):
        if x < self.threshold:
            return self.children[0]
        return self.children[1]


def test_size():

    tree = BinaryBranch(
        BinaryBranch(
            BinaryBranch(
                BinaryBranch(Leaf(), Leaf()),
                Leaf()
            ),
            BinaryBranch(Leaf(), Leaf()),
        ),
        BinaryBranch(Leaf(), Leaf()),
    )

    assert (
        tree.n_nodes ==
        tree.n_branches + tree.n_leaves ==
        6 + 7
    )
    assert (
        tree.children[0].n_nodes ==
        tree.children[0].n_branches + tree.children[0].n_leaves ==
        4 + 5
    )
    assert (
        tree.children[1].n_nodes ==
        tree.children[1].n_branches + tree.children[1].n_leaves ==
        1 + 2
    )
    assert (
        tree.children[1].children[0].n_nodes ==
        tree.children[1].children[0].n_branches + tree.children[1].children[0].n_leaves ==
        0 + 1
    )


def test_height():

    tree = BinaryBranch(
        BinaryBranch(
            BinaryBranch(
                BinaryBranch(Leaf(), Leaf()),
                Leaf(),
            ),
            BinaryBranch(Leaf(), Leaf()),
        ),
        BinaryBranch(Leaf(), Leaf()),
    )

    assert tree.height == 5
    assert tree.children[0].height == 4
    assert tree.children[1].height == 2
    assert tree.children[1].children[0].height == 1


def test_iter_dfs():

    tree = BinaryBranch(
        BinaryBranch(
            Leaf(no=3),
            Leaf(no=4),
            no=2
        ),
        Leaf(no=5),
        no=1
    )

    for i, node in enumerate(tree.iter_dfs(), start=1):
        assert i == node.no


def iter_leaves():

    tree = BinaryBranch(
        BinaryBranch(
            Leaf(no=1),
            Leaf(no=2),
        ),
        Leaf(no=3)
    )

    for i, leaf in enumerate(tree.iter_leaves(), start=1):
        assert i == leaf.no
