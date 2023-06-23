from __future__ import annotations

from . import base


class SMSSpam(base.RemoteDataset):
    """SMS Spam Collection dataset.

    The data contains 5,574 items and 1 feature (i.e. SMS body). Spam messages represent
    13.4% of the dataset. The goal is to predict whether an SMS is a spam or not.

    References
    ----------
    [^1]: [Almeida, T.A., Hidalgo, J.M.G. and Yamakami, A., 2011, September. Contributions to the study of SMS spam filtering: new collection and results. In Proceedings of the 11th ACM symposium on Document engineering (pp. 259-262).](http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/doceng11.pdf)

    """

    def __init__(self):
        super().__init__(
            n_samples=5_574,
            n_features=1,
            task=base.BINARY_CLF,
            url="https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip",
            size=477_907,
            filename="SMSSpamCollection",
        )

    def _iter(self):
        with open(self.path) as f:
            for row in f:
                label, body = row.split("\t")
                yield ({"body": body}, label == "spam")
