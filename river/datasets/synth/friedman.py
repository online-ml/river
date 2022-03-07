import math
import random
from typing import Tuple

from .. import base


class Friedman(base.SyntheticDataset):
    """Friedman synthetic dataset.

    Each observation is composed of 10 features. Each feature value is sampled uniformly in [0, 1].
    The target is defined by the following function:

    $$y = 10 sin(\\pi x_0 x_1) + 20 (x_2 - 0.5)^2 + 10 x_3 + 5 x_4 + \\epsilon$$

    In the last expression, $\\epsilon \\sim \\mathcal{N}(0, 1)$, is the noise. Therefore,
    only the first 5 features are relevant.

    Parameters
    ----------
    seed
        Random seed number used for reproducibility.

    Examples
    --------

    >>> from river import synth

    >>> dataset = synth.Friedman(seed=42)

    >>> for x, y in dataset.take(5):
    ...     print(list(x.values()), y)
    [0.63, 0.02, 0.27, 0.22, 0.73, 0.67, 0.89, 0.08, 0.42, 0.02] 7.66
    [0.02, 0.19, 0.64, 0.54, 0.22, 0.58, 0.80, 0.00, 0.80, 0.69] 8.33
    [0.34, 0.15, 0.95, 0.33, 0.09, 0.09, 0.84, 0.60, 0.80, 0.72] 7.04
    [0.37, 0.55, 0.82, 0.61, 0.86, 0.57, 0.70, 0.04, 0.22, 0.28] 18.16
    [0.07, 0.23, 0.10, 0.27, 0.63, 0.36, 0.37, 0.20, 0.26, 0.93] 8.90

    References
    ----------
    [^1]: [Friedman, J.H., 1991. Multivariate adaptive regression splines. The annals of statistics, pp.1-67.](https://projecteuclid.org/euclid.aos/1176347963)

    """

    def __init__(self, seed: int = None):
        super().__init__(task=base.REG, n_features=10)
        self.seed = seed

    def __iter__(self):

        rng = random.Random(self.seed)

        while True:

            x = {i: rng.uniform(a=0, b=1) for i in range(10)}
            y = (
                10 * math.sin(math.pi * x[0] * x[1])
                + 20 * (x[2] - 0.5) ** 2
                + 10 * x[3]
                + 5 * x[4]
                + rng.gauss(mu=0, sigma=1)
            )

            yield x, y


class FriedmanDrift(Friedman):
    """Friedman synthetic dataset with concept drifts.

    Each observation is composed of 10 features. Each feature value is sampled uniformly in [0, 1].
    Only the first 5 features are relevant. The target is defined by different functions depending
    on the type of the drift.

    The three available modes of operation of the data generator are described in [^1].

    Parameters
    ----------
    drift_type
        The variant of concept drift.</br>
        - `'lea'`: Local Expanding Abrupt drift. The concept drift appears in two distinct
        regions of the instance space, while the remaining regions are left unaltered.
        There are three points of abrupt change in the training dataset.
        At every consecutive change the regions of drift are expanded.</br>
        - `'gra'`: Global Recurring Abrupt drift. The concept drift appears over the whole
        instance space. There are two points of concept drift. At the second point of drift
        the old concept reoccurs.</br>
        - `'gsg'`: Global and Slow Gradual drift. The concept drift affects all the instance
        space. However, the change is gradual and not abrupt. After each one of the two change
        points covered by this variant, and during a window of length `transition_window`,
        examples from both old and the new concepts are generated with equal probability.
        After the transition period, only the examples from the new concept are generated.
    position
        The amount of monitored instances after which each concept drift occurs. A tuple with
        at least two element must be passed, where each number is greater than the preceding one.
        If `drift_type='lea'`, then the tuple must have three elements.
    transition_window
        The length of the transition window between two concepts. Only applicable when
         `drift_type='gsg'`. If set to zero, the drifts will be abrupt. Anytime
         `transition_window > 0`, it defines a window in which instances of the new
         concept are gradually introduced among the examples from the old concept.
         During this transition phase, both old and new concepts appear with equal probability.
    seed
        Random seed number used for reproducibility.

    Examples
    --------
    >>> from river import synth

    >>> dataset = synth.FriedmanDrift(
    ...     drift_type='lea',
    ...     position=(1, 2, 3),
    ...     seed=42
    ... )

    >>> for x, y in dataset.take(5):
    ...     print(list(x.values()), y)
    [0.63, 0.02, 0.27, 0.22, 0.73, 0.67, 0.89, 0.08, 0.42, 0.02] 7.66
    [0.02, 0.19, 0.64, 0.54, 0.22, 0.58, 0.80, 0.00, 0.80, 0.69] 8.33
    [0.34, 0.15, 0.95, 0.33, 0.09, 0.09, 0.84, 0.60, 0.80, 0.72] 7.04
    [0.37, 0.55, 0.82, 0.61, 0.86, 0.57, 0.70, 0.04, 0.22, 0.28] 18.16
    [0.07, 0.23, 0.10, 0.27, 0.63, 0.36, 0.37, 0.20, 0.26, 0.93] -2.65

    >>> dataset = synth.FriedmanDrift(
    ...     drift_type='gra',
    ...     position=(2, 3),
    ...     seed=42
    ... )

    >>> for x, y in dataset.take(5):
    ...     print(list(x.values()), y)
    [0.63, 0.02, 0.27, 0.22, 0.73, 0.67, 0.89, 0.08, 0.42, 0.02] 7.66
    [0.02, 0.19, 0.64, 0.54, 0.22, 0.58, 0.80, 0.00, 0.80, 0.69] 8.33
    [0.34, 0.15, 0.95, 0.33, 0.09, 0.09, 0.84, 0.60, 0.80, 0.72] 8.96
    [0.37, 0.55, 0.82, 0.61, 0.86, 0.57, 0.70, 0.04, 0.22, 0.28] 18.16
    [0.07, 0.23, 0.10, 0.27, 0.63, 0.36, 0.37, 0.20, 0.26, 0.93] 8.90

    >>> dataset = synth.FriedmanDrift(
    ...     drift_type='gsg',
    ...     position=(1, 4),
    ...     transition_window=2,
    ...     seed=42
    ... )

    >>> for x, y in dataset.take(5):
    ...     print(list(x.values()), y)
    [0.63, 0.02, 0.27, 0.22, 0.73, 0.67, 0.89, 0.08, 0.42, 0.02] 7.66
    [0.02, 0.19, 0.64, 0.54, 0.22, 0.58, 0.80, 0.00, 0.80, 0.69] 8.33
    [0.34, 0.15, 0.95, 0.33, 0.09, 0.09, 0.84, 0.60, 0.80, 0.72] 8.92
    [0.37, 0.55, 0.82, 0.61, 0.86, 0.57, 0.70, 0.04, 0.22, 0.28] 17.32
    [0.07, 0.23, 0.10, 0.27, 0.63, 0.36, 0.37, 0.20, 0.26, 0.93] 6.05

    References
    ----------
    [^1]: Ikonomovska, E., Gama, J. and Džeroski, S., 2011. Learning model trees from evolving
    data streams. Data mining and knowledge discovery, 23(1), pp.128-168.

    """

    _LOCAL_EXPANDING_ABRUPT = "lea"
    _GLOBAL_RECURRING_ABRUPT = "gra"
    _GLOBAL_AND_SLOW_GRADUAL = "gsg"

    _VALID_DRIFT_TYPES = [
        _LOCAL_EXPANDING_ABRUPT,
        _GLOBAL_RECURRING_ABRUPT,
        _GLOBAL_AND_SLOW_GRADUAL,
    ]

    def __init__(
        self,
        drift_type: str = "lea",
        position: Tuple[int, ...] = (50_000, 100_000, 150_000),
        transition_window: int = 10_000,
        seed: int = None,
    ):
        super().__init__(seed=seed)

        if drift_type not in self._VALID_DRIFT_TYPES:
            raise ValueError(
                f'Invalid "drift_type: {drift_type}"\n'
                f"Valid options are: {self._VALID_DRIFT_TYPES}"
            )

        self.drift_type = drift_type

        if self.drift_type == self._LOCAL_EXPANDING_ABRUPT and len(position) < 3:
            raise ValueError(
                "Insufficient number of concept drift locations passed.\n"
                'Three concept drift points should be passed when drift_type=="lea"'
            )
        elif self.drift_type != self._LOCAL_EXPANDING_ABRUPT and len(position) < 2:
            raise ValueError(
                "Insufficient number of concept drift locations passed.\n"
                "Two locations must be defined."
            )
        elif len(position) > 3:
            raise ValueError(
                "Too many concept drift locations passed. Check the documentation"
                "for details on the usage of this class."
            )

        self.position = position

        if self.drift_type == self._LOCAL_EXPANDING_ABRUPT:
            (
                self._change_point1,
                self._change_point2,
                self._change_point3,
            ) = self.position
        else:
            self._change_point1, self._change_point2 = self.position
            self._change_point3 = math.inf

        if not self._change_point1 < self._change_point2 < self._change_point3:
            raise ValueError(
                "The concept drift locations must be defined in an increasing order."
            )

        if (
            transition_window > self._change_point2 - self._change_point1
            or transition_window > self._change_point3 - self._change_point2
        ) and self.drift_type == self._GLOBAL_AND_SLOW_GRADUAL:
            raise ValueError(
                f'The chosen "transition_window" value is too big: {transition_window}'
            )

        self.transition_window = transition_window

        if self.drift_type == self._LOCAL_EXPANDING_ABRUPT:
            self._y_maker = self._local_expanding_abrupt_gen
        elif self.drift_type == self._GLOBAL_RECURRING_ABRUPT:
            self._y_maker = self._global_recurring_abrupt_gen
        else:  # Global and slow gradual drifts
            self._y_maker = self._global_and_slow_gradual_gen

    def __lea_in_r1(self, x, index):
        if index < self._change_point1:
            return False
        elif self._change_point1 <= index < self._change_point2:
            return x[1] < 0.3 and x[2] < 0.3 and x[3] > 0.7 and x[4] < 0.3
        elif self._change_point2 <= index < self._change_point3:
            return x[1] < 0.3 and x[2] < 0.3 and x[3] > 0.7
        else:
            return x[1] < 0.3 and x[2] < 0.3

    def __lea_in_r2(self, x, index):
        if index < self._change_point1:
            return False
        elif self._change_point1 <= index < self._change_point2:
            return x[1] > 0.7 and x[2] > 0.7 and x[3] < 0.3 and x[4] > 0.7
        elif self._change_point2 <= index < self._change_point3:
            return x[1] > 0.7 and x[2] > 0.7 and x[3] < 0.3
        else:
            return x[1] > 0.7 and x[2] > 0.7

    def _local_expanding_abrupt_gen(
        self, x, index: int, rc: random.Random = None
    ):  # noqa
        if self.__lea_in_r1(x, index):
            return 10 * x[0] * x[1] + 20 * (x[2] - 0.5) + 10 * x[3] + 5 * x[4]

        if self.__lea_in_r2(x, index):
            return (
                10 * math.cos(x[0] * x[1])
                + 20 * (x[2] - 0.5)
                + math.exp(x[3])
                + 5 * x[4] ** 2
            )

        # default case
        return (
            10 * math.sin(math.pi * x[0] * x[1])
            + 20 * (x[2] - 0.5) ** 2
            + 10 * x[3]
            + 5 * x[4]
        )

    def _global_recurring_abrupt_gen(
        self, x, index: int, rc: random.Random = None
    ):  # noqa
        if index < self._change_point1 or index >= self._change_point2:
            # The initial concept is recurring
            return (
                10 * math.sin(math.pi * x[0] * x[1])
                + 20 * (x[2] - 0.5) ** 2
                + 10 * x[3]
                + 5 * x[4]
            )
        else:
            # Drift: the positions of the features are swapped
            return (
                10 * math.sin(math.pi * x[3] * x[5])
                + 20 * (x[1] - 0.5) ** 2
                + 10 * x[0]
                + 5 * x[2]
            )

    def _global_and_slow_gradual_gen(self, x, index: int, rc: random.Random):
        if index < self._change_point1:
            # default function
            return (
                10 * math.sin(math.pi * x[0] * x[1])
                + 20 * (x[2] - 0.5) ** 2
                + 10 * x[3]
                + 5 * x[4]
            )
        elif self._change_point1 <= index < self._change_point2:
            if index < self._change_point1 + self.transition_window and bool(
                rc.getrandbits(1)
            ):
                # default function
                return (
                    10 * math.sin(math.pi * x[0] * x[1])
                    + 20 * (x[2] - 0.5) ** 2
                    + 10 * x[3]
                    + 5 * x[4]
                )
            else:  # First new function
                return (
                    10 * math.sin(math.pi * x[3] * x[4])
                    + 20 * (x[1] - 0.5) ** 2
                    + 10 * x[0]
                    + 5 * x[2]
                )
        elif index >= self._change_point2:
            if index < self._change_point2 + self.transition_window and bool(
                rc.getrandbits(1)
            ):
                # First new function
                return (
                    10 * math.sin(math.pi * x[3] * x[4])
                    + 20 * (x[1] - 0.5) ** 2
                    + 10 * x[0]
                    + 5 * x[2]
                )
            else:  # Second new function
                return (
                    10 * math.sin(math.pi * x[1] * x[4])
                    + 20 * (x[3] - 0.5) ** 2
                    + 10 * x[2]
                    + 5 * x[0]
                )

    def __iter__(self):
        rng = random.Random(self.seed)

        # To produce True or False with equal probability. Only used in gradual drifts
        if self.drift_type == self._GLOBAL_AND_SLOW_GRADUAL:
            rc = random.Random(self.seed)
        else:
            rc = None

        i = 0
        while True:
            x = {i: rng.uniform(a=0, b=1) for i in range(10)}
            y = self._y_maker(x, i, rc) + rng.gauss(mu=0, sigma=1)

            yield x, y
            i += 1
