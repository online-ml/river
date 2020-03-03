from .. import stream

from . import base


class Taxis(base.FileDataset):
    """Taxi ride durations in New York City.

    The goal is to predict the duration of taxi rides in New York City.

    Parameters:
        data_home (str): The directory where you wish to store the data.
        verbose (bool): Whether to indicate download progress or not.

    Yields:
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.

    References:
        1. `New York City Taxi Trip Duration competition on Kaggle <https://www.kaggle.com/c/nyc-taxi-trip-duration>`_

    """

    def __init__(self, data_home=None, verbose=True):
        super().__init__(
            n_samples=1_458_644,
            n_features=8,
            category=base.REG,
            url='https://maxhalford.github.io/files/datasets/nyc_taxis.zip',
            data_home=data_home,
            verbose=verbose
        )

    def _stream_X_y(self, directory):
        return stream.iter_csv(
            f'{directory}/train.csv',
            target_name='trip_duration',
            converters={
                'passenger_count': int,
                'pickup_longitude': float,
                'pickup_latitude': float,
                'dropoff_longitude': float,
                'dropoff_latitude': float,
                'trip_duration': int
            },
            parse_dates={'pickup_datetime': '%Y-%m-%d %H:%M:%S'},
            drop=['dropoff_datetime', 'id']
        )
