class FoundNode(object):
    """ Base class for tree nodes.

    Parameters
    ----------
    node: SplitNode or LearningNode
        The node object.
    parent: SplitNode or None
        The node's parent.
    parent_branch: int
        The parent node's branch.

    """

    def __init__(self, node=None, parent=None, parent_branch=None):
        """ FoundNode class constructor. """
        self.node = node
        self.parent = parent
        self.parent_branch = parent_branch
