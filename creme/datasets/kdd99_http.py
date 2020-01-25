from .. import stream

from . import base


class KDD99HTTP(base.Dataset):
    """Data from the HTTP dataset of the KDD 1999 cup.

    The goal is to predict whether or not an HTTP connection is anomalous or not. The dataset only contains 2,211 (0.4%) positive labels.

    Parameters:
        data_home (str): The directory where you wish to store the data.
        verbose (bool): Whether to indicate download progress or not.

    Yields:
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.

    References:
        1. `HTTP (KDDCUP99) dataset <http://odds.cs.stonybrook.edu/http-kddcup99-dataset/>`_

    """

    def __init__(self, data_home=None, verbose=True):
        super().__init__(
            n_samples=567_498,
            n_features=3,
            category=base.BINARY_CLF,
            url='https://maxhalford.github.io/files/datasets/kdd99_http.zip',
            data_home=data_home,
            verbose=verbose
        )

    def _stream_X_y(self, directory):
        return stream.iter_csv(
            f'{directory}/kdd99_http.csv',
            target_name='service',
            converters={
                'duration': float,
                'src_bytes': float,
                'dst_bytes': float,
                'service': int
            }
        )
