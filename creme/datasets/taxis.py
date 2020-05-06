from creme import stream

from . import base


class Taxis(base.FileDataset):
    """Taxi ride durations in New York City.

    The goal is to predict the duration of taxi rides in New York City.

    Parameters:
        data_home: The directory where you wish to store the data.
        verbose: Whether to indicate download progress or not.

    References:
        1. [New York City Taxi Trip Duration competition on Kaggle](https://www.kaggle.com/c/nyc-taxi-trip-duration)

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
            target='trip_duration',
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
