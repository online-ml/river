import os

from creme import stream

from . import base


class Bananas(base.FileDataset):
    """Bananas dataset.

    An artificial dataset where instances belongs to several clusters with a banana shape.
    There are two attributes that correspond to the x and y axis, respectively.

    References:
        1. [OpenML page](https://www.openml.org/d/1460)

    """

    def __init__(self):
        super().__init__(n_samples=5300, n_features=2, category=base.BINARY_CLF)

    def _stream_X_y(self, directory):
        return stream.iter_libsvm(
            os.path.join(directory, 'banana.zip'),
            target_type=lambda x: x == '1'
        )
