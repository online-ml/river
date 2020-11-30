import numpy as np

from .. import base
from river.utils.skmultiflow_utils import check_random_state


class Mixed(base.SyntheticDataset):
    r"""Mixed data stream generator.

    This generator is an implementation of a data stream with abrupt concept
    drift and boolean noise-free examples as described in [^1].

    It has four relevant attributes, two boolean attributes $v, w$ and two
    numeric attributes $x, y$ uniformly distributed from 0 to 1. The examples
    are labeled depending on the classification function chosen from below.

    * `function 0`:
      if $v$ and $w$ are true or $v$ and $z$ are true or $w$ and $z$ are true
      then 0 else 1, where $z$ is $y < 0.5 + 0.3 sin(3 \pi  x)$

    * `function 1`:
       The opposite of `function 0`.

    Concept drift can be introduced by changing the classification function.
    This can be done manually or using `ConceptDriftStream`.

    Parameters
    ----------
    classification_function
        Which of the two classification functions to use for the generation.
        Valid options are 0 or 1.
    seed
        If int, `seed` is used to seed the random number generator;
        If RandomState instance, `seed` is the random number generator;
        If None, the random number generator is the `RandomState` instance used
        by `np.random`.
    balance_classes
        Whether to balance classes or not. If balanced, the class distribution
        will converge to a uniform distribution.

    Examples
    --------
    >>> from river import synth
    >>>
    >>> dataset = synth.Mixed(seed = 42, classification_function=1, balance_classes = True)
    >>>
    >>> for x, y in dataset.take(5):
    ...     print(x, y)
    {0: False, 1: True, 2: 0.7319, 3: 0.5986} 1
    {0: False, 1: False, 2: 0.0580, 3: 0.8661} 0
    {0: True, 1: True, 2: 0.0205, 3: 0.9699} 1
    {0: False, 1: True, 2: 0.4319, 3: 0.2912} 0
    {0: True, 1: False, 2: 0.2921, 3: 0.3663} 1

    Notes
    -----
    The sample generation works as follows: The two numeric attributes are
    generated with the random  generator initialized with the seed passed by
    the user (optional). The boolean attributes are either 0 or 1
    based on the comparison of the random number generator and 0.5 ,
    the classification function decides whether to classify the instance
    as class 0 or class 1. The next step is to verify if the classes should
    be balanced, and if so, balance the classes.

    The generated sample will have 4 relevant features and 1 label (it is a
    binary-classification task).

    References
    ----------
    [^1]: Gama, Joao, et al. "Learning with drift detection." Advances in
          artificial intelligence–SBIA 2004. Springer Berlin Heidelberg,
          2004. 286-295"

    """

    def __init__(
        self,
        classification_function: int = 0,
        seed: int or np.random.RandomState = None,
        balance_classes: bool = False,
    ):
        super().__init__(n_features=4, n_classes=2, n_outputs=1, task=base.BINARY_CLF)

        # Classification functions to use
        self._functions = [
            self._classification_function_zero,
            self._classification_function_one,
        ]
        self.seed = seed
        if classification_function not in [0, 1]:
            raise ValueError(
                f"Invalid classification_function ({classification_function}). "
                "Valid values are 0 or 1."
            )
        self.classification_function = classification_function
        self._rng = None  # This is the actual random_state object used internally
        self.balance_classes = balance_classes
        self.n_cat_features = 2
        self.n_num_features = 2
        self.cat_features_idx = [0, 1]
        self.next_class_should_be_zero = False

    def __iter__(self):
        self._rng = check_random_state(self.seed)
        self.next_class_should_be_zero = False

        while True:
            att_0 = False
            att_1 = False
            att_2 = 0.0
            att_3 = 0.0
            y = 0
            desired_class_found = False
            while not desired_class_found:
                att_0 = self._rng.rand() >= 0.5
                att_1 = self._rng.rand() >= 0.5
                att_2 = self._rng.rand()
                att_3 = self._rng.rand()

                y = self._functions[self.classification_function](att_0, att_1, att_2, att_3)

                if not self.balance_classes:
                    desired_class_found = True
                else:
                    if (self.next_class_should_be_zero and (y == 0)) or (
                        (not self.next_class_should_be_zero) and (y == 1)
                    ):
                        desired_class_found = True
                        self.next_class_should_be_zero = not self.next_class_should_be_zero

            x = {0: att_0, 1: att_1, 2: att_2, 3: att_3}

            yield x, y

    @staticmethod
    def _classification_function_zero(v: bool, w: bool, x: float, y: float):
        z = y < 0.5 + 0.3 * np.sin(3 * np.pi * x)
        return 0 if (v and w) or (v and z) or (w and z) else 1

    @staticmethod
    def _classification_function_one(v: bool, w: bool, x: float, y: float):
        z = y < 0.5 + 0.3 * np.sin(3 * np.pi * x)
        return 1 if (v == 1 and w == 1) or (v == 1 and z) or (w == 1 and z) else 0

    def generate_drift(self):
        """Generate drift by switching the classification function."""
        self.classification_function = 1 - self.classification_function
