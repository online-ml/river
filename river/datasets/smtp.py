from __future__ import annotations

from river import stream

from . import base


class SMTP(base.RemoteDataset):
    """SMTP dataset from the KDD 1999 cup.

    The goal is to predict whether or not an SMTP connection is anomalous or not. The dataset only
    contains 30 (0.03%) positive labels.

    References
    ----------
    [^1]: [KDD Cup 1999 dataset](https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)

    """

    def __init__(self):
        super().__init__(
            n_samples=95_156,
            n_features=3,
            task=base.BINARY_CLF,
            url="https://maxhalford.github.io/files/datasets/smtp.zip",
            size=5_484_982,
            filename="smtp.csv",
        )

    def _iter(self):
        return stream.iter_csv(
            self.path,
            target="service",
            converters={
                "duration": float,
                "src_bytes": float,
                "dst_bytes": float,
                "service": int,
            },
        )
