from __future__ import annotations

from river import stream

from . import base


class TREC07(base.RemoteDataset):
    """TREC's 2007 Spam Track dataset.

    The data contains 75,419 chronologically ordered items, i.e. 3 months of emails delivered
    to a particular server in 2007. Spam messages represent 66.6% of the dataset.
    The goal is to predict whether an email is a spam or not.

    The available raw features are: sender, recipients, date, subject, body.

    References
    ----------
    [^1]: [TREC 2007 Spam Track Overview](https://trec.nist.gov/pubs/trec16/papers/SPAM.OVERVIEW16.pdf)
    [^2]: [Code ran to parse the dataset](https://gist.github.com/gbolmier/b6a942699aaaedec54041a32e4f34d40)

    """

    def __init__(self):
        super().__init__(
            n_samples=75_419,
            n_features=5,
            task=base.BINARY_CLF,
            url="https://maxhalford.github.io/files/datasets/trec07p.zip",
            size=144_504_829,
            filename="trec07p.csv",
        )

    def _iter(self):
        return stream.iter_csv(
            self.path,
            target="y",
            delimiter=",",
            quotechar='"',
            field_size_limit=1_000_000,
        )
