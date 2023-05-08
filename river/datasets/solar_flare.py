from __future__ import annotations

from river import stream

from . import base


class SolarFlare(base.FileDataset):
    """Solar flare multi-output regression.

    References
    ----------
    [^1]: [UCI page](https://archive.ics.uci.edu/ml/datasets/Solar+Flare)

    """

    def __init__(self):
        super().__init__(
            n_samples=1_066,
            n_features=10,
            n_outputs=3,
            task=base.MO_REG,
            filename="solar-flare.csv.zip",
        )

    def __iter__(self):
        return stream.iter_csv(
            self.path,
            target=["c-class-flares", "m-class-flares", "x-class-flares"],
            converters={
                "zurich-class": str,
                "largest-spot-size": str,
                "spot-distribution": str,
                "activity": int,
                "evolution": int,
                "previous-24h-flare-activity": int,
                "hist-complex": int,
                "hist-complex-this-pass": int,
                "area": int,
                "largest-spot-area": int,
                "c-class-flares": int,
                "m-class-flares": int,
                "x-class-flares": int,
            },
        )
