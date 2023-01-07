import random
from abc import ABC


class MondrianTree(ABC):
    """
    Base class for Mondrian Trees.

    This is an **abstract class**, so it cannot be used directly. It defines base operations
    and properties that all the Mondrian Trees must inherit or implement according to
    their own design.

    Parameters
    ----------
    n_features
        Number of features
    step
        Step parameter of the tree
    loss
        Loss to minimize for each node of the tree
        Pick between: "log", ...
    use_aggregation
        Whether or not the tree should it use aggregation
    split_pure
        Whether or not the tree should split pure leaves when training
    iteration
        Number of iterations to run when training
    seed
        Random seed for reproducibility
    """

    def __init__(
        self,
        n_features: int,
        step: float = 0.1,
        loss: str = "log",
        use_aggregation: bool = True,
        split_pure: bool = False,
        iteration: int = 0,
        seed: int = None,
    ):
        # Properties common to all the Mondrian Trees
        self.n_features = n_features
        self.step = step
        self.loss = loss
        self.use_aggregation = use_aggregation
        self.split_pure = split_pure
        self.iteration = iteration
        self.intensities = [0.0 for _ in range(n_features)]
        self.random_generator = random.Random(seed)

        # One should initialize the tree in the child class as well
