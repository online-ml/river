from __future__ import annotations

from river import stream

from . import base


class Higgs(base.RemoteDataset):
    """Higgs dataset.

    The data has been produced using Monte Carlo simulations. The first 21 features (columns 2-22)
    are kinematic properties measured by the particle detectors in the accelerator. The last seven
    features are functions of the first 21 features; these are high-level features derived by
    physicists to help discriminate between the two classes.

    References
    ----------
    [^1]: [UCI page](https://archive.ics.uci.edu/ml/datasets/HIGGS)

    """

    def __init__(self):
        super().__init__(
            n_samples=11_000_000,
            n_features=28,
            task=base.BINARY_CLF,
            url="https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz",
            size=2_816_407_858,
            unpack=False,
        )

    def _iter(self):
        features = [
            "lepton pT",
            "lepton eta",
            "lepton phi",
            "missing energy magnitude",
            "missing energy phi",
            "jet 1 pt",
            "jet 1 eta",
            "jet 1 phi",
            "jet 1 b-tag",
            "jet 2 pt",
            "jet 2 eta",
            "jet 2 phi",
            "jet 2 b-tag",
            "jet 3 pt",
            "jet 3 eta",
            "jet 3 phi",
            "jet 3 b-tag",
            "jet 4 pt",
            "jet 4 eta",
            "jet 4 phi",
            "jet 4 b-tag",
            "m_jj",
            "m_jjj",
            "m_lv",
            "m_jlv",
            "m_bb",
            "m_wbb",
            "m_wwbb",
        ]

        return stream.iter_csv(
            self.path,
            fieldnames=["is_signal", *features],
            target="is_signal",
            converters={
                "is_signal": lambda x: x.startswith("1"),
                **{f: float for f in features},
            },
        )
