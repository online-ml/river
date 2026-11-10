from __future__ import annotations

import io
import zipfile

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
        super().__init__(filename="banana.zip", n_samples=5300, n_features=2, task=base.BINARY_CLF)

    def __iter__(self):
        with (
            zipfile.ZipFile(self.path) as archive,
            archive.open("banana.all.txt") as member,
            io.TextIOWrapper(member, encoding="utf-8") as buffer,
        ):
            yield from stream.iter_libsvm(buffer, target_type=lambda x: x == "1")
