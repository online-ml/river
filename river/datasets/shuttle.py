from __future__ import annotations

from river import stream

from . import base


class Shuttle(base.FileDataset):
    """Statlog (Shuttle) dataset.

    This is the UCI Statlog (Shuttle) dataset, cast as a binary anomaly-detection problem following
    the ODDS benchmark. The nine numerical attributes are sensor readings from a NASA space shuttle.
    The original seven classes are collapsed into a normal class (the majority "Rad Flow" class) and
    an anomaly class (the rare classes), while the second-largest class is discarded. The result is
    49,097 observations of which 3,511 (~7%) are anomalies.

    The target `anomaly` is `1` for an anomaly and `0` otherwise.

    References
    ----------
    [^1]: [UCI page](https://archive.ics.uci.edu/dataset/148/statlog+shuttle)
    [^2]: [ODDS page](https://odds.cs.stonybrook.edu/shuttle-dataset/)

    """

    def __init__(self) -> None:
        super().__init__(
            n_samples=49_097,
            n_features=9,
            task=base.BINARY_CLF,
            filename="shuttle.csv.gz",
        )

    def __iter__(self):
        return stream.iter_csv(
            self.path,
            target="anomaly",
            converters={f"f{i}": int for i in range(1, 10)} | {"anomaly": int},
        )
