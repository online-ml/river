from river import stream

from . import base


class HTTP(base.RemoteDataset):
    """HTTP dataset of the KDD 1999 cup.

    The goal is to predict whether or not an HTTP connection is anomalous or not. The dataset only
    contains 2,211 (0.4%) positive labels.

    References
    ----------
    [^1]: [HTTP (KDDCUP99) dataset](http://odds.cs.stonybrook.edu/http-kddcup99-dataset/)

    """

    def __init__(self):
        super().__init__(
            n_samples=567_498,
            n_features=3,
            task=base.BINARY_CLF,
            url="https://maxhalford.github.io/files/datasets/kdd99_http.zip",
            size=32400738,
            filename="kdd99_http.csv",
        )

    def _iter(self):
        converters = {
            "duration": float,
            "src_bytes": float,
            "dst_bytes": float,
            "service": int,
        }
        return stream.iter_csv(self.path, target="service", converters=converters)
