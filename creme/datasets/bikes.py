from . import base


def fetch_bikes(data_home=None):
    """Bike sharing station information from the city of Toulouse.

    Parameters:
        data_home (str): The directory where you wish to store the data.

    Yields:
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.

    References:
        1. `A short introduction and conclusion to the OpenBikes 2016 Challenge <https://maxhalford.github.io/blog/a-short-introduction-and-conclusion-to-the-openbikes-2016-challenge/>`_

    """

    name = 'toulouse_bikes'
    url = 'https://maxhalford.github.io/files/datasets/toulouse_bikes.zip'

    return base.fetch_csv_dataset(
        data_home=data_home,
        url=url,
        name=name,
        target_name='bikes',
        types={
            'clouds': int,
            'humidity': int,
            'pressure': float,
            'temperature': float,
            'wind': float,
            'bikes': int
        },
        parse_dates={'moment': '%Y-%m-%d %H:%M:%S'}
    )
