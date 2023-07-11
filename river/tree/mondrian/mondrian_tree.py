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
    seed
        Random seed for reproducibility.

    """

    def __init__(
        self,
        step: float = 0.1,
        loss: str = "log",
        use_aggregation: bool = True,
        iteration: int = 0,
        seed: int | None = None,
    ):
        # Properties common to all the Mondrian Trees
        self.step = step
        self.loss = loss
        self.use_aggregation = use_aggregation
        self.iteration = iteration

        # Controls the randomness in the tree
        self.seed = seed
        self._rng = random.Random(seed)
