from . import base


class SMS(base.Dataset):
    """SMS Spam Collection dataset.

    The data contains 5,574 items and 1 feature (i.e. SMS body). Spam messages represent
    13.4% of the dataset. The goal is to predict whether an SMS is a spam or not.

    Parameters:
        data_home (str): The directory where you wish to store the data.
        verbose (bool): Whether to indicate download progress or not.

    Yields:
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.

    References:
        1. `Contributions to the Study of SMS Spam Filtering: New Collection and Results <http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/doceng11.pdf>`_

    """

    def __init__(self, data_home=None, verbose=False):
        super().__init__(
            n_samples=5_574,
            n_features=1,
            category=base.BINARY_CLF,
            name='sms_spam',
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip',
            data_home=data_home,
            archive_type='zip',
            verbose=verbose
        )

    def _stream_X_y(self, directory):
        with open(f'{directory}/SMSSpamCollection') as f:
            for ix, row in enumerate(f):
                label, body = row.split('\t')
                yield ({'body': body}, label == 'spam')
