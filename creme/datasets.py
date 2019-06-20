import os
import shutil
import urllib
import zipfile

from . import stream


__all__ = [
    'fetch_bikes',
    'fetch_electricity',
    'fetch_restaurants',
    'load_airline',
    'load_chick_weights'
]


def get_data_home(data_home=None):
    """Return the path of the creme data directory."""
    if data_home is None:
        data_home = os.environ.get('CREME_DATA', os.path.join('~', 'creme_data'))
        data_home = os.path.expanduser(data_home)
        if not os.path.exists(data_home):
            os.makedirs(data_home)
    return data_home


def fetch_csv_dataset(data_home, url, name, silent=True, **iter_csv_params):

    data_home = get_data_home(data_home=data_home)

    # If the CSV file exists then iterate over it
    csv_path = os.path.join(data_home, f'{name}.csv')
    if os.path.exists(csv_path):
        return stream.iter_csv(csv_path, **iter_csv_params)

    # If the ZIP file exists then unzip it
    zip_path = os.path.join(data_home, f'{name}.zip')
    if os.path.exists(zip_path):

        # Unzip the ZIP file
        if not silent:
            print('Unzipping data...')
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(data_home)

        # Delete the ZIP file now that the CSV file is available
        os.remove(zip_path)

        return fetch_csv_dataset(data_home, url, name, **iter_csv_params)

    # Download the ZIP file
    if not silent:
        print('Downloading data...')
    with urllib.request.urlopen(url) as r, open(zip_path, 'wb') as f:
        shutil.copyfileobj(r, f)

    return fetch_csv_dataset(data_home, url, name, **iter_csv_params)


def fetch_bikes(data_home=None, silent=True):
    """Bike sharing station information from the city of Toulouse.

    The data contains 182,470 items and 8 features. The goal is to predict the number of bikes in
    5 different bike stations from the city of Toulouse.

    Parameters:
        data_home (str): The directory where you wish to store the data.
        silent (bool): Whether to indicate download progress or not.

    Yields:
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.

    References:
        1. `A short introduction and conclusion to the OpenBikes 2016 Challenge <https://maxhalford.github.io/blog/a-short-introduction-and-conclusion-to-the-openbikes-2016-challenge/>`_

    """

    name = 'toulouse_bikes'
    url = 'https://maxhalford.github.io/files/datasets/toulouse_bikes.zip'

    return fetch_csv_dataset(
        data_home=data_home,
        url=url,
        name=name,
        silent=silent,
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


def fetch_electricity(data_home=None, silent=True):
    """A day of electricity prices in New South Wales.

    The data contains 45,312 items and 8 features. The goal is to predict whether the price of
    electricity will go up or down at very frequent intervals.

    Parameters:
        data_home (str): The directory where you wish to store the data.
        silent (bool): Whether to indicate download progress or not.

    Yields:
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.

    References:
        1. `SPLICE-2 Comparative Evaluation: Electricity Pricing <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.12.9405>`_

    """

    name = 'electricity'
    url = 'https://maxhalford.github.io/files/datasets/electricity.zip'

    return fetch_csv_dataset(
        data_home=data_home,
        url=url,
        name=name,
        silent=silent,
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


def fetch_restaurants(data_home=None, silent=True):
    """Data from the Kaggle Recruit Restaurants challenge.

    The data contains 252,108 items and 7 features. The goal is to predict the number of visitors
    in each of 829 Japanese restaurants over a priod of roughly 16 weeks. The data is ordered by
    date and then by restaurant ID.

    Parameters:
        data_home (str): The directory where you wish to store the data.
        silent (bool): Whether to indicate download progress or not.

    Yields:
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.

    References:
        1. `Recruit Restaurant Visitor Forecasting <https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting>`_

    """

    name = 'kaggle_recruit_restaurants'
    url = 'https://maxhalford.github.io/files/datasets/kaggle_recruit_restaurants.zip'

    return fetch_csv_dataset(
        data_home=data_home,
        url=url,
        name=name,
        silent=silent,
        target_name='visitors',
        types={
            'latitude': float,
            'longitude': float,
            'visitors': int,
            'is_holiday': bool
        },
        parse_dates={'date': '%Y-%m-%d'}
    )


def load_airline():
    """Monthly number of international airline passengers.

    The stream contains 144 items and only one single feature, which is the month. The goal is to
    predict the number of passengers each month by capturing the trend and the seasonality of the
    data.

    Yields:
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.

    References:
        1. `International airline passengers: monthly totals in thousands. Jan 49 – Dec 60 <https://datamarket.com/data/set/22u3/international-airline-passengers-monthly-totals-in-thousands-jan-49-dec-60#!ds=22u3&display=line>`_

    """
    return stream.iter_csv(
        os.path.join(os.path.dirname(__file__), 'data', 'airline-passengers.csv'),
        target_name='passengers',
        types={'passengers': int},
        parse_dates={'month': '%Y-%m'}
    )


def load_chick_weights():
    """Chick weights along time.

    The stream contains 578 items and 3 features. The goal is to predict the weight of each chick
    along time, according to the diet the chick is on. The data is ordered by time and then by
    chick.

    Yields:
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.

    References:
        1. `Chick Weight <http://rstudio-pubs-static.s3.amazonaws.com/107631_131ad1c022df4f90aa2d214a5c5609b2.html>`_

    """
    return stream.iter_csv(
        os.path.join(os.path.dirname(__file__), 'data', 'chick-weights.csv'),
        target_name='weight',
        types={'time': int, 'weight': int, 'chick': int, 'diet': int}
    )
