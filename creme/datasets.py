import ast
import os
import shutil
import tarfile
import urllib
import zipfile

from . import stream


__all__ = [
    'fetch_bikes',
    'fetch_electricity',
    'fetch_kdd99_http',
    'fetch_restaurants',
    'fetch_sms',
    'fetch_trec07p',
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


def download_dataset(name, url, data_home, archive_type=None, silent=True):
    """Downloads/decompresses a dataset locally if does not exist.

    Parameters:
        name (str): Dataset filename or directory name.
        url (str): From where to download the dataset.
        data_home (str): The directory where you wish to store the data.
        archive_type (str): Dataset archive type extension name (e.g. 'zip'). Defaults to None.
        silent (bool): Whether to indicate download progress or not.

    Returns:
        data_dir_path (str): Where the dataset is stored.

    """
    data_home = get_data_home(data_home=data_home)
    data_dir_path = os.path.join(data_home, f'{name}')

    # Download dataset if does not exist
    if not os.path.exists(data_dir_path):
        if not silent:
            print('Downloading data...')

        if archive_type:
            data_dir_path = f'{data_dir_path}.{archive_type}'

        with urllib.request.urlopen(url) as r, open(data_dir_path, 'wb') as f:
            shutil.copyfileobj(r, f)

        # Uncompress if needed
        if archive_type:
            archive_path, data_dir_path = data_dir_path, data_dir_path[:-len(archive_type) - 1]

            if not silent:
                print('Uncompressing data...')

            if archive_type == 'zip':
                with zipfile.ZipFile(archive_path, 'r') as zf:
                    zf.extractall(data_dir_path)
            elif archive_type in ['gz', 'tar', 'tar.gz', 'tgz']:
                mode = 'r:' if archive_type == 'tar' else 'r:gz'
                tar = tarfile.open(archive_path, mode)
                tar.extractall(data_dir_path)
                tar.close()

            # Delete the archive file now that the dataset is available
            os.remove(archive_path)

    return data_dir_path


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

    # Download dataset if does not exist and get its path
    data_dir_path = download_dataset(name, url, data_home, archive_type='zip', silent=silent)

    return stream.iter_csv(
        f'{data_dir_path}/{name}.csv',
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

    # Download dataset if does not exist and get its path
    data_dir_path = download_dataset(name, url, data_home, archive_type='zip', silent=silent)

    return stream.iter_csv(
        f'{data_dir_path}/{name}.csv',
        target_name='class',
        converters={
            'date': float,
            'day': int,
            'period': float,
            'nswprice': float,
            'nswdemand': float,
            'vicprice': float,
            'vicdemand': float,
            'transfer': float,
            'class': lambda x: x == 'UP'
        }
    )


def fetch_kdd99_http(data_home=None, silent=True):
    """Data from the HTTP dataset of the KDD'99 cup.

    The data contains 567,498 items and 3 features. The goal is to predict whether or not an HTTP
    connection is anomalous or not. The dataset only contains 2,211 (0.4%) positive labels.

    Parameters:
        data_home (str): The directory where you wish to store the data.
        silent (bool): Whether to indicate download progress or not.

    Yields:
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.

    References:
        1. `HTTP (KDDCUP99) dataset <http://odds.cs.stonybrook.edu/http-kddcup99-dataset/>`_

    """

    name = 'kdd99_http'
    url = 'https://maxhalford.github.io/files/datasets/kdd99_http.zip'

    # Download dataset if does not exist and get its path
    data_dir_path = download_dataset(name, url, data_home, archive_type='zip', silent=silent)

    return stream.iter_csv(
        f'{data_dir_path}/{name}.csv',
        target_name='service',
        converters={
            'duration': float,
            'src_bytes': float,
            'dst_bytes': float,
            'service': int
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

    # Download dataset if does not exist and get its path
    data_dir_path = download_dataset(name, url, data_home, archive_type='zip', silent=silent)

    return stream.iter_csv(
        f'{data_dir_path}/{name}.csv',
        target_name='visitors',
        converters={
            'latitude': float,
            'longitude': float,
            'visitors': int,
            'is_holiday': ast.literal_eval
        },
        parse_dates={'date': '%Y-%m-%d'}
    )


def fetch_sms(data_home=None, silent=True):
    """SMS Spam Collection dataset.

    The data contains 5,574 items and 1 feature (i.e. sms body). Spam messages represent
    13.4% of the dataset. The goal is to predict whether a sms is a spam or not.

    Parameters:
        data_home (str): The directory where you wish to store the data.
        silent (bool): Whether to indicate download progress or not.

    Yields:
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.

    References:
        1. `Contributions to the Study of SMS Spam Filtering: New Collection and Results <http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/doceng11.pdf>`_

    """

    name = 'SMSSpamCollection'
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'

    # Download dataset if does not exist and get its path
    data_dir_path = download_dataset(name, url, data_home, archive_type='zip', silent=silent)

    # Stream sms
    with open(f'{data_dir_path}/{name}') as f:
        for ix, row in enumerate(f):
            label, body = row.split('\t')
            yield ({'body': body}, label)


def fetch_trec07p(data_home=None, silent=True):
    """2007 TREC’s Spam Track dataset.

    The data contains 75,419 chronologically ordered items, i.e. 3 months of emails delivered
    to a particular server in 2007. Spam messages represent 66.6% of the dataset.
    The goal is to predict whether an email is a spam or not.

    Parsed features are: sender, recipients, date, subject, body.

    Parameters:
        data_home (str): The directory where you wish to store the data.
        silent (bool): Whether to indicate download progress or not.

    Yields:
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.

    References:
        1. `TREC 2007 Spam Track Overview <https://trec.nist.gov/pubs/trec16/papers/SPAM.OVERVIEW16.pdf>`_
        2. `Code ran to parse the dataset <https://gist.github.com/gbolmier/b6a942699aaaedec54041a32e4f34d40>`_

    """

    name = 'trec07p'
    url = 'https://maxhalford.github.io/files/datasets/trec07p.zip'

    # Download dataset if does not exist andCode ran to build trec07p CSV available in `creme` library get its path
    data_dir_path = download_dataset(name, url, data_home, archive_type='zip', silent=silent)

    return stream.iter_csv(f'{data_dir_path}/{name}.csv', target_name='y',)


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
        converters={'passengers': int},
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
        converters={'time': int, 'weight': int, 'chick': int, 'diet': int}
    )
