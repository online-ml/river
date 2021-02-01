from river import stream

from . import base


class Bananas(base.FileDataset):
    """Bananas dataset.

    An artificial dataset where instances belongs to several clusters with a banana shape.
    There are two attributes that correspond to the x and y axis, respectively.

    References
    ----------
    [^1]: [OpenML page](https://www.openml.org/d/1460)

    """

    def __init__(self):
        super().__init__(
            filename="banana.zip", n_samples=5300, n_features=2, task=base.BINARY_CLF
        )

    def __iter__(self):
        return stream.iter_libsvm(self.path, target_type=lambda x: x == "1")
