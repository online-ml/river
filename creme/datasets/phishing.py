import os

from .. import stream

from . import base


class Phishing(base.Dataset):
    """Phishing websites.

    This dataset contains features from web pages that are classified as phishing or not.

    Yields:
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.

    References:
        1. `UCI page <http://archive.ics.uci.edu/ml/datasets/Website+Phishing>`_

    """

    def __init__(self):
        super().__init__(
            n_samples=1250,
            n_features=9,
            category=base.BINARY_CLF
        )

    def _stream_X_y(self, directory):
        return stream.iter_csv(
            os.path.join(directory, 'phishing.csv.gz'),
            target_name='is_phishing',
            converters={
                'empty_server_form_handler': int,
                'popup_window': int,
                'https': int,
                'request_from_other_domain': int,
                'anchor_from_other_domain': int,
                'is_popular': int,
                'long_url': int,
                'age_of_domain': int,
                'ip_in_url': int,
                'is_phishing': int
            }
        )
