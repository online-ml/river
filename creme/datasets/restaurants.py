import ast

from .. import stream

from . import base


class Restaurants(base.FileDataset):
    """Data from the Kaggle Recruit Restaurants challenge.

    The goal is to predict the number of visitors in each of 829 Japanese restaurants over a priod
    of roughly 16 weeks. The data is ordered by date and then by restaurant ID.

    Parameters:
        data_home: The directory where you wish to store the data.
        verbose: Whether to indicate download progress or not.

    References:
        1. [Recruit Restaurant Visitor Forecasting](https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting)

    """

    def __init__(self, data_home: str = None, verbose=True):
        super().__init__(
            n_samples=252_108,
            n_features=7,
            category=base.REG,
            url='https://maxhalford.github.io/files/datasets/kaggle_recruit_restaurants.zip',
            data_home=data_home,
            verbose=verbose
        )

    def _stream_X_y(self, directory):
        return stream.iter_csv(
            f'{directory}/kaggle_recruit_restaurants.csv',
            target='visitors',
            converters={
                'latitude': float,
                'longitude': float,
                'visitors': int,
                'is_holiday': ast.literal_eval
            },
            parse_dates={'date': '%Y-%m-%d'}
        )
