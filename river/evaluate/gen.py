from __future__ import annotations

from river import datasets


class WrappedGenerator(datasets.base.Dataset):
    def __init__(self, n_samples, n_features, task, gen):
        super().__init__(
            n_samples=n_samples,
            n_features=n_features,
            task=task,
        )

        self._gen = gen

    def __iter__(self):
        for i, (x, y) in enumerate(self._gen):
            if i == self.n_samples:
                break
            yield x, y


class Friedman7k(WrappedGenerator):
    """Sample from the stationary version of the Friedman dataset.

    This sample contains 10k instances sampled from the Friedman generator.

    """

    def __init__(self):
        super().__init__(
            n_samples=7000,
            n_features=10,
            task=datasets.base.REG,
            gen=datasets.synth.Friedman(seed=42),
        )


class FriedmanLEA10k(WrappedGenerator):
    """Sample from the FriedmanLEA generator.

    This sample contains 10k instances sampled from the Friedman generator and presents
    local-expanding abrupt concept drifts that locally affect the data and happen after
    2k, 5k, and 8k instances.

    """

    def __init__(self):
        super().__init__(
            n_samples=10000,
            n_features=10,
            task=datasets.base.REG,
            gen=datasets.synth.FriedmanDrift(
                drift_type="lea", position=(2000, 5000, 8000), seed=42
            ),
        )


class FriedmanGSG10k(WrappedGenerator):
    """Sample from the FriedmanGSG generator.

    This sample contains 10k instances sampled from the Friedman generator and presents
    global and slow gradual concept drifts that affect the data and happen after
    3.5k and 7k instances. The transition window between different concepts has a length of
    1k instances.

    """

    def __init__(self):
        super().__init__(
            n_samples=10000,
            n_features=10,
            task=datasets.base.REG,
            gen=datasets.synth.FriedmanDrift(
                drift_type="gsg", position=(3500, 7000), transition_window=1000, seed=42
            ),
        )
