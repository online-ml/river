from __future__ import annotations

import os

from . import base


class MaliciousURL(base.RemoteDataset):
    """Malicious URLs dataset.

    This dataset contains features about URLs that are classified as malicious or not.

    References
    ----------
    [^1]: [Detecting Malicious URLs](http://www.sysnet.ucsd.edu/projects/url/)
    [^2]: [Identifying Suspicious URLs: An Application of Large-Scale Online Learning](http://cseweb.ucsd.edu/~jtma/papers/url-icml2009.pdf)

    """

    def __init__(self):
        super().__init__(
            n_samples=2_396_130,
            n_features=3_231_961,
            task=base.BINARY_CLF,
            url="http://www.sysnet.ucsd.edu/projects/url/url_svmlight.tar.gz",
            filename="url_svmlight",
            size=2_210_273_352,
            sparse=True,
        )

    def _iter(self):
        files = list(self.path.glob("Day*.svm"))
        files.sort(key=lambda x: int(os.path.basename(x).split(".")[0][3:]))

        def parse_libsvm_feature(f):
            k, v = f.split(":")
            return int(k), float(v)

        # There are 150 files with each one corresponding to a day
        for file in files:
            with open(file) as f:
                for line in f:
                    # Each file has the libsvm format
                    elements = line.rstrip().split(" ")
                    y = elements.pop(0) == "+1"
                    x = dict(parse_libsvm_feature(f) for f in elements)
                    yield x, y
