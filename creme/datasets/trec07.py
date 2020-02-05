from .. import stream

from . import base


class TREC07(base.FileDataset):
    """TREC's 2007 Spam Track dataset.

    The data contains 75,419 chronologically ordered items, i.e. 3 months of emails delivered
    to a particular server in 2007. Spam messages represent 66.6% of the dataset.
    The goal is to predict whether an email is a spam or not.

    The available raw features are: sender, recipients, date, subject, body.

    Parameters:
        data_home (str): The directory where you wish to store the data.
        verbose (bool): Whether to indicate download progress or not.

    Yields:
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.

    References:
        .. [1] `TREC 2007 Spam Track Overview <https://trec.nist.gov/pubs/trec16/papers/SPAM.OVERVIEW16.pdf>`_
        .. [2] `Code ran to parse the dataset <https://gist.github.com/gbolmier/b6a942699aaaedec54041a32e4f34d40>`_

    """

    def __init__(self, data_home=None, verbose=True):
        super().__init__(
            n_samples=75_419,
            n_features=5,
            category=base.BINARY_CLF,
            url='https://maxhalford.github.io/files/datasets/trec07p.zip',
            data_home=data_home,
            verbose=verbose
        )

    def _stream_X_y(self, directory):
        return stream.iter_csv(
            f'{directory}/trec07p.csv',
            target_name='y',
            delimiter=',',
            quotechar='"',
            field_size_limit=1_000_000,
        )
