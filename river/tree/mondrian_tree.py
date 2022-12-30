from abc import ABC
from river.tree.nodes.mondrian_tree_nodes import MondrianTreeBranch


class MondrianTree(ABC):
    """
    Base class for Mondrian Trees.

    This is an **abstract class**, so it cannot be used directly. It defines base operations
    and properties that all the Mondrian Trees must inherit or implement according to
    their own design.

    Parameters
    ----------
        n_features: int,
            Number of features
        step: float,
            Step parameter of the tree
        loss: str
            Default is "log"
        use_aggregation: bool,
            Should it use aggregation
        split_pure: bool,
            Should the tree split pure leafs when training
        iteration: int,
            Number of iterations to run when training

    Attributes
    ----------
        intensities: list[float]
            List of intensity per feature
        tree: MondrianTreeBranch or None
            Base branch of the tree (starting root)
    """

    def __init__(
            self,
            n_features: int,
            step: float,
            loss,
            use_aggregation: bool,
            split_pure: bool,
            iteration: int,
    ):
        # Properties common to all the Mondrian Trees
        self.n_features = n_features
        self.step = step
        self.loss = loss
        self.use_aggregation = use_aggregation
        self.split_pure = split_pure
        self.iteration = iteration
        self.intensities = [0] * n_features
        self.tree = None
