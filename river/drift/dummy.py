from __future__ import annotations

import math
import random

from river import base


class DummyDriftDetector(base.DriftDetector):
    """Baseline drift detector that generates pseudo drift detection signals.

    There are two approaches[^1]:

    - `fixed` where the drift signal is generated every `t_0` samples.
    - `random` corresponds to a pseudo-random drift detection strategy.

    Parameters
    ----------
    trigger_method
        The trigger method to use.<br/>
        * `fixed`<br/>
        * `random`
    t_0
        Reference point to define triggers.
    w
        Auxiliary parameter whose purpose is twofold:</br>
        - if `trigger_method="fixed"`, the periodic drift signals will only start after an initial
        warm-up period randomly defined between `[0, w]`. Useful to avoid that all ensemble
        members are reset at the same time when periodic triggers are used as the adaptation strategy.</br>
        - if `trigger_method="random"`, `w` defines the probability bounds of triggering a drift. The
        chance of triggering a drift is $0.5$ after observing `t_0` instances and becomes $1$ after
        monitoring `t_0 + w / 2` instances. A sigmoid function is used to produce values between `[0, 1]`
        that are used as the reset probabilities.
    dynamic_cloning
        Whether to change the `seed` and `w` values each time `clone()` is called.
    seed
        Random seed for reproducibility.

    Examples
    --------
    >>> import random
    >>> from river import drift

    >>> rng = random.Random(42)

    The observed values will not affect the periodic triggers.

    >>> data = [rng.gauss(0, 1) for _ in range(1000)]

    Let's start with the fixed drift signals:

    >>> ptrigger = DummyDriftDetector(t_0=500, seed=42)
    >>> for i, v in enumerate(data):
    ...     _ = ptrigger.update(v)
    ...     if ptrigger.drift_detected:
    ...         print(f"Drift detected at instance {i}.")
    Drift detected at instance 499.
    Drift detected at instance 999.

    Now, the random drift signals:

    >>> rtrigger = DummyDriftDetector(
    ...     trigger_method="random",
    ...     t_0=500,
    ...     w=100,
    ...     dynamic_cloning=True,
    ...     seed=42
    ... )
    >>> for i, v in enumerate(data):
    ...     _ = rtrigger.update(v)
    ...     if rtrigger.drift_detected:
    ...         print(f"Drift detected at instance {i}.")
    Drift detected at instance 368.
    Drift detected at instance 817.

    Remember to set a w > 0 value if random triggers are used:

    >>> try:
    ...     DummyDriftDetector(trigger_method="random")
    ... except ValueError as ve:
    ...     print(ve)
    The 'w' value must be greater than zero when 'trigger_method' is 'random'.

    Since we set `dynamic_cloning` to `True`, a clone of the periodic trigger will
    have its internal paramenters changed:

    >>> rtrigger = rtrigger.clone()
    >>> for i, v in enumerate(data):
    ...     _ = rtrigger.update(v)
    ...     if rtrigger.drift_detected:
    ...         print(f"Drift detected at instance {i}.")
    Drift detected at instance 429.
    Drift detected at instance 728.

    Notes
    -----
    When used in ensembles, a naive implementation of periodic drift signals would make all ensemble members
    reset at the same time. To avoid that, the `dynamic_cloning` parameter can be set to `True`. In this case,
    every time the `clone` method of this detector is called in an ensemble a new `seed` is defined. If
    `dynamic_cloning=True` and `trigger_method="fixed"`, a new `w` between `[0, t_0]` will also be created
    for the new cloned instance.

    References
    ----------
    [^1]: Heitor Gomes, Jacob Montiel, Saulo Martiello Mastelini, Bernhard Pfahringer, and Albert Bifet.
    On Ensemble Techniques for Data Stream Regression. IJCNN'20. International Joint Conference on
    Neural Networks. 2020.

    """

    _FIXED_TRIGGER = "fixed"
    _RANDOM_TRIGGER = "random"

    def __init__(
        self,
        trigger_method: str = "fixed",
        t_0: int = 300,
        w: int = 0,
        dynamic_cloning: bool = False,
        seed: int | None = None,
    ):
        super().__init__()
        if trigger_method not in {self._FIXED_TRIGGER, self._RANDOM_TRIGGER}:
            raise ValueError(
                f"Invalid trigger_method: {trigger_method}.\n"
                f"Valid options are: {[self._FIXED_TRIGGER, self._RANDOM_TRIGGER]}"
            )
        self.trigger_method = trigger_method

        if self.trigger_method == self._RANDOM_TRIGGER and w == 0:
            raise ValueError(
                "The 'w' value must be greater than zero when 'trigger_method' is 'random'."
            )

        self.t_0 = t_0
        self.w = w
        self.dynamic_cloning = dynamic_cloning
        self.seed = seed

        self._rng = random.Random(self.seed)
        self._n = 0

        if self.trigger_method == self._FIXED_TRIGGER:
            if self.w > 0:
                self._warmup = self._rng.randint(1, self.w + 1)
                self._warmup_done = False
            else:
                self._warmup_done = True

        if self.trigger_method == self._FIXED_TRIGGER:
            self._trigger = self._fixed_trigger
        else:  # self.trigger_method == self._RANDOM_TRIGGER
            self._trigger = self._random_trigger

    def _fixed_trigger(self):
        if not self._warmup_done:
            if self._n >= self._warmup:
                self._n = 0
                self._warmup_done = True
        elif self._n >= self.t_0:
            self._drift_detected = True
            self._n = 0

    def _random_trigger(self):
        t = self._n
        t_0 = self.t_0
        threshold = 1 / (1 + math.exp(-4 * (t - t_0) / self.w))
        self._drift_detected = self._rng.random() < threshold

        if self.drift_detected:
            self._n = 0

    def update(self, x):
        self._n += 1
        self._drift_detected = False
        self._trigger()

        return self

    def clone(self):
        new = (
            super().clone(
                {
                    "seed": self._rng.randint(0, int(1e15)),
                    "w": (
                        self._rng.randint(0, self.t_0 + 1)
                        if self.trigger_method == self._FIXED_TRIGGER
                        else self.w
                    ),
                }
            )
            if self.dynamic_cloning
            else super().clone()
        )

        return new
