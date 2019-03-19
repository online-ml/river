from . import base


def fetch_restaurants(data_home=None):
    """Returns a stream containing the data from the Kaggle Recruit Restaurants challenge.

    The data is ordered by data and then by store ID.

    Parameters:
        data_home (str): The directory where you wish to store the data.

    Yields:
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.

    1. `Recruit Restaurant Visitor Forecasting <https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting>`_

    """

    name = 'kaggle_recruit_restaurants'
    url = 'https://maxhalford.github.io/files/datasets/kaggle_recruit_restaurants.zip'

    return base.fetch_csv_dataset(
        data_home=data_home,
        url=url,
        name=name,
        target_name='visitors',
        types={
            'latitude': float,
            'longitude': float,
            'visitors': int,
            'is_holiday': bool
        },
        parse_dates={'date': '%Y-%m-%d'}
    )
