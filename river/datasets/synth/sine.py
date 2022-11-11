from __future__ import annotations

import numpy as np

from river import datasets
from river.utils.skmultiflow_utils import check_random_state


class Sine(datasets.base.SyntheticDataset):
    r"""Sine generator.

    This generator is an implementation of the dara stream with abrupt
    concept drift, as described in Gama, Joao, et al. [^1].

    It generates up to 4 relevant numerical features, that vary from 0 to 1,
    where only 2 of them are relevant to the classification task and the other
    2 are optionally added by as noise. A classification function is chosen
    among four options:

    0. `SINE1`. Abrupt concept drift, noise-free examples. It has two relevant
       attributes. Each attributes has values uniformly distributed in [0, 1].
       In the first context all points below the curve $y = sin(x)$ are
       classified as positive.

    1. `Reversed SINE1`. The reversed classification of `SINE1`.

    2. `SINE2`. The same two relevant attributes. The classification function
       is $y < 0.5 + 0.3 sin(3 \pi  x)$.

    3. `Reversed SINE2`. The reversed classification of `SINE2`.

    Concept drift can be introduced by changing the classification function.
    This can be done manually or using `ConceptDriftStream`.

    Two important features are the possibility to balance classes, which means
    the class distribution will tend to a uniform one, and the possibility
    to add noise, which will, add two non relevant attributes.

    Parameters
    ----------
    classification_function
        Classification functions to use. From 0 to 3.
    seed
        If int, `seed` is used to seed the random number generator;
        If RandomState instance, `seed` is the random number generator;
        If None, the random number generator is the `RandomState` instance used
        by `np.random`.
    balance_classes
        Whether to balance classes or not. If balanced, the class
        distribution will converge to an uniform distribution.
    has_noise
        Adds 2 non relevant features to the stream.

    Notes
    -----
    The sample generation works as follows: The two attributes are
    generated with the random number generator. The classification function
    defines whether to classify the instance as class 0 or class 1. Finally,
    data is balanced and noise is added, if these options are set by the user.

    The generated sample will have 2 relevant features, and an additional
    two noise features if `has_noise` is set.

    Examples
    --------
    >>> from river.datasets import synth

    >>> dataset = synth.Sine(classification_function = 2, seed = 112,
    ...                      balance_classes = False, has_noise = True)

    >>> for x, y in dataset.take(5):
    ...     print(x, y)
    {0: 0.3750, 1: 0.6403, 2: 0.9500, 3: 0.0756} 1
    {0: 0.7769, 1: 0.8327, 2: 0.0548, 3: 0.8176} 1
    {0: 0.8853, 1: 0.7223, 2: 0.0025, 3: 0.9811} 0
    {0: 0.3434, 1: 0.0947, 2: 0.3946, 3: 0.0049} 1
    {0: 0.7367, 1: 0.9558, 2: 0.8206, 3: 0.3449} 0

    References
    ----------
    [^1]: Gama, Joao, et al.'s 'Learning with drift detection.'
          Advances in artificial intelligence–SBIA 2004.
          Springer Berlin Heidelberg, 2004. 286-295."

    """
    _N_BASE_FEATURES = 2
    _N_FEATURES_INCLUDING_NOISE = 4

    def __init__(
        self,
        classification_function: int = 0,
        seed: int | np.random.RandomState | None = None,
        balance_classes: bool = False,
        has_noise: bool = False,
    ):
        super().__init__(
            n_features=self._N_BASE_FEATURES if not has_noise else self._N_FEATURES_INCLUDING_NOISE,
            n_classes=2,
            n_outputs=1,
            task=datasets.base.BINARY_CLF,
        )

        # Classification functions to use
        self._functions = [
            self._classification_function_zero,
            self._classification_function_one,
            self._classification_function_two,
            self._classification_function_three,
        ]
        if classification_function not in range(4):
            raise ValueError(
                f"Invalid classification_function {classification_function}. "
                "Valid values are: 0, 1, 2, 3."
            )
        self.classification_function = classification_function
        self.seed = seed
        self.has_noise = has_noise
        self.balance_classes = balance_classes
        self._rng = None  # This is the actual random_state object used internally
        self.next_class_should_be_zero = False
        self.target_values = [i for i in range(self.n_classes)]

    def __iter__(self):
        self._rng = check_random_state(self.seed)
        self.next_class_should_be_zero = False

        while True:
            x = dict()
            y = 0
            desired_class_found = False
            while not desired_class_found:
                x[0] = self._rng.rand()
                x[1] = self._rng.rand()
                y = self._functions[self.classification_function](x[0], x[1])

                if not self.balance_classes:
                    desired_class_found = True
                else:
                    if (self.next_class_should_be_zero and (y == 0)) or (
                        (not self.next_class_should_be_zero) and (y == 1)
                    ):
                        desired_class_found = True
                        self.next_class_should_be_zero = not self.next_class_should_be_zero

            if self.has_noise:
                x[2] = self._rng.rand()
                x[3] = self._rng.rand()

            yield x, y

    def generate_drift(self):
        """Generate drift by switching the classification function at random."""
        new_function = self._rng.randint(4)
        while new_function == self.classification_function:
            new_function = self._rng.randint(4)
        self.classification_function = new_function

    @staticmethod
    def _classification_function_zero(att1, att2):
        # SINE1 function
        return 0 if (att1 >= np.sin(att2)) else 1

    @staticmethod
    def _classification_function_one(att1, att2):
        # Reversed SINE1 function
        return 0 if (att1 < np.sin(att2)) else 1

    @staticmethod
    def _classification_function_two(att1, att2):
        # SINE2 function
        return 0 if (att1 >= 0.5 + 0.3 * np.sin(3 * np.pi * att2)) else 1

    @staticmethod
    def _classification_function_three(att1, att2):
        # Reversed SINE2 function
        return 0 if (att1 < 0.5 + 0.3 * np.sin(3 * np.pi * att2)) else 1
