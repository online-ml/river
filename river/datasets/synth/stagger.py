from __future__ import annotations

import random

from river import datasets


class STAGGER(datasets.base.SyntheticDataset):
    """STAGGER concepts stream generator.

    This generator is an implementation of the dara stream with abrupt concept
    drift, as described in [^1].

    The STAGGER concepts are boolean functions `f` with three features
    describing objects: size (small, medium and large), shape (circle, square
    and triangle) and colour (red, blue and green).

    `f` options:

    0. `True` if the size is small and the color is red.

    1. `True` if the color is green or the shape is a circle.

    2. `True` if the size is medium or large

    Concept drift can be introduced by changing the classification function.
    This can be done manually or using `datasets.synth.ConceptDriftStream`.

    One important feature is the possibility to balance classes, which
    means the class distribution will tend to a uniform one.

    Parameters
    ----------
    classification_function
        Classification functions to use. From 0 to 2.
    seed
        Random seed for reproducibility.
    balance_classes
        Whether to balance classes or not. If balanced, the class
        distribution will converge to an uniform distribution.

    Examples
    --------
    >>> from river.datasets import synth

    >>> dataset = synth.STAGGER(classification_function = 2, seed = 112,
    ...                      balance_classes = False)

    >>> for x, y in dataset.take(5):
    ...     print(x, y)
    {'size': 1, 'color': 2, 'shape': 2} 1
    {'size': 2, 'color': 1, 'shape': 2} 1
    {'size': 1, 'color': 1, 'shape': 2} 1
    {'size': 0, 'color': 1, 'shape': 0} 0
    {'size': 2, 'color': 1, 'shape': 0} 1

    Notes
    -----
    The sample generation works as follows: The 3 attributes are
    generated with the random number generator. The classification function
    defines whether to classify the instance as class 0 or class 1. Finally,
    data is balanced, if this option is set by the user.

    References
    ----------
    [^1]: Schlimmer, J. C., & Granger, R. H. (1986). Incremental learning
          from noisy data. Machine learning, 1(3), 317-354.

    """

    def __init__(
        self,
        classification_function: int = 0,
        seed: int | None = None,
        balance_classes: bool = False,
    ):
        super().__init__(n_features=3, n_classes=2, n_outputs=1, task=datasets.base.BINARY_CLF)

        # Classification functions to use
        self._functions = [
            self._classification_function_zero,
            self._classification_function_one,
            self._classification_function_two,
        ]
        if classification_function not in range(3):
            raise ValueError(
                f"Invalid classification_function {classification_function}. "
                "Valid values are: 0, 1, 2."
            )
        self.classification_function = classification_function
        self.seed = seed
        self.balance_classes = balance_classes
        self.n_cat_features = 3
        self._rng = None  # This is the actual random_state object used internally
        self.next_class_should_be_zero = False

        self.feature_names = ["size", "color", "shape"]
        self.size_labels = {0: "small", 1: "medium", 2: "large"}
        self.color_labels = {0: "red", 1: "blue", 2: "green"}
        self.shape_labels = {0: "circle", 1: "square", 2: "triangle"}
        self.target_values = [i for i in range(self.n_classes)]

    def __iter__(self):
        self._rng = random.Random(self.seed)
        self.next_class_should_be_zero = False

        while True:
            size = 0
            color = 0
            shape = 0
            y = 0
            desired_class_found = False
            while not desired_class_found:
                size = self._rng.randint(0, 2)
                color = self._rng.randint(0, 2)
                shape = self._rng.randint(0, 2)

                y = self._functions[self.classification_function](size, color, shape)

                if not self.balance_classes:
                    desired_class_found = True
                else:
                    if (self.next_class_should_be_zero and (y == 0)) or (
                        (not self.next_class_should_be_zero) and (y == 1)
                    ):
                        desired_class_found = True
                        self.next_class_should_be_zero = not self.next_class_should_be_zero

            x = {"size": size, "color": color, "shape": shape}

            yield x, y

    def generate_drift(self):
        """Generate drift by switching the classification function at random."""
        new_function = self._rng.randint(0, 2)
        while new_function == self.classification_function:
            new_function = self._rng.randint(0, 2)
        self.classification_function = new_function

    @staticmethod
    def _classification_function_zero(size, color, shape):
        # Class label 1 if the color is red and size is small.
        return 1 if (size == 0 and color == 0) else 0

    @staticmethod
    def _classification_function_one(size, color, shape):
        # Class label 1 if the color is green or shape is a circle.
        return 1 if (color == 2 or shape == 0) else 0

    @staticmethod
    def _classification_function_two(size, color, shape):
        # Class label 1 if the size is medium or large.
        return 1 if (size == 1 or size == 2) else 0
