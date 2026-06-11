from __future__ import annotations

import abc
import random


class MondrianTree(abc.ABC):
    """Base class for Mondrian Trees.

    This is an **abstract class**, so it cannot be used directly. It defines base operations
    and properties that all the Mondrian Trees must inherit or implement according to
    their own design.

    Parameters
    ----------
    step
        Step parameter of the tree.
    loss
        Loss to minimize for each node of the tree. At the moment it is a placeholder.
        In the future, different optimization metrics might become available.
    use_aggregation
        Whether or not the tree should it use aggregation.
    iteration
        Number of iterations to run when training.
    max_nodes
        Maximum number of nodes allowed in the tree. No new splits will occur once this
        limit is reached. If `None`, the tree grows without bound. Setting this limits
        memory usage at the cost of potentially less accurate predictions.
    seed
        Random seed for reproducibility.

    """

    def __init__(
        self,
        step: float = 1.0,
        loss: str = "log",
        use_aggregation: bool = True,
        iteration: int = 0,
        max_nodes: int | None = None,
        seed: int | None = None,
    ):
        # Properties common to all the Mondrian Trees
        self.step = step
        self.loss = loss
        self.use_aggregation = use_aggregation
        self.iteration = iteration
        self.max_nodes = max_nodes

        # Number of nodes currently in the tree (starts at 1 for the root)
        self._n_nodes = 1

        # Controls the randomness in the tree
        self.seed = seed
        self._rng = random.Random(seed)
