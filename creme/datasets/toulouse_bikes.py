from .. import stream

from . import base


class ToulouseBikes(base.FileDataset):
    """Bike sharing station information from the city of Toulouse.

    The goal is to predict the number of bikes in 5 different bike stations from the city of
    Toulouse.

    Parameters:
        data_home (str): The directory where you wish to store the data.
        verbose (bool): Whether to indicate download progress or not.

    Yields:
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.

    References:
        .. [1] `A short introduction and conclusion to the OpenBikes 2016 Challenge <https://maxhalford.github.io/blog/a-short-introduction-and-conclusion-to-the-openbikes-2016-challenge/>`_

    """

    def __init__(self, data_home=None, verbose=False):
        super().__init__(
            n_samples=182_470,
            n_features=8,
            category=base.REG,
            url='https://maxhalford.github.io/files/datasets/toulouse_bikes.zip',
            data_home=data_home,
            verbose=verbose
        )

    def _stream_X_y(self, directory):
        return stream.iter_csv(
            f'{directory}/toulouse_bikes.csv',
            target_name='bikes',
            converters={
                'clouds': int,
                'humidity': int,
                'pressure': float,
                'temperature': float,
                'wind': float,
                'bikes': int
            },
            parse_dates={'moment': '%Y-%m-%d %H:%M:%S'}
        )
