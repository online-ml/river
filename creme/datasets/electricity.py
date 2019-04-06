from . import base


def fetch_electricity(data_home=None):
    """Returns a stream containing a day of electricity prices in New South Wales.

    Parameters:
        data_home (str): The directory where you wish to store the data.

    Yields:
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.

    1. `SPLICE-2 Comparative Evaluation: Electricity Pricing <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.12.9405>`_

    """

    name = 'electricity'
    url = 'https://maxhalford.github.io/files/datasets/electricity.zip'

    return base.fetch_csv_dataset(
        data_home=data_home,
        url=url,
        name=name,
        target_name='class',
        types={
            'date': float,
            'day': int,
            'period': float,
            'nswprice': float,
            'nswdemand': float,
            'vicprice': float,
            'vicdemand': float,
            'transfer': float,
            'class': lambda x: True if x == 'UP' else False
        }
    )
