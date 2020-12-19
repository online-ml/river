from river import stream

from . import base


class SMTP(base.RemoteDataset):
    """SMTP dataset from the KDD 1999 cup.

    The goal is to predict whether or not an SMTP connection is anomalous or not. The dataset only
    contains 2,211 (0.4%) positive labels.

    References
    ----------
    [^1]: [SMTP (KDDCUP99) dataset](http://odds.cs.stonybrook.edu/smtp-kddcup99-dataset/)

    """

    def __init__(self):
        super().__init__(
            n_samples=95_156,
            n_features=3,
            task=base.BINARY_CLF,
            url="https://maxhalford.github.io/files/datasets/smtp.zip",
            size=5484982,
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
