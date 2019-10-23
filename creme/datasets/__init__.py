"""Toy datasets."""
import ast
import os
import shutil
import tarfile
import urllib
import zipfile

from .. import stream


__all__ = [
    'fetch_bikes',
    'fetch_credit_card',
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
            print(f'Downloading {name}...')

        if archive_type:
            data_dir_path = f'{data_dir_path}.{archive_type}'

        with urllib.request.urlopen(url) as r, open(data_dir_path, 'wb') as f:
            shutil.copyfileobj(r, f)

        # Uncompress if needed
        if archive_type:
            archive_path, data_dir_path = data_dir_path, data_dir_path[:-len(archive_type) - 1]

            if not silent:
                print(f'Uncompressing {name}...')

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

    This data was collected from the Australian New South Wales Electricity Market. In this market,
    prices are not fixed and are affected by demand and supply of the market. They are set every
    five minutes. Electricity transfers to/from the neighboring state of Victoria were done to
    alleviate fluctuations.

    The dataset (originally named ELEC2) contains 45,312 instances dated from 7 May 1996 to 5
    December 1998. Each example of the dataset refers to a period of 30 minutes, i.e. there are 48
    instances for each time period of one day. Each example on the dataset has 5 fields, the day of
    week, the time stamp, the New South Wales electricity demand, the Victoria electricity demand,
    the scheduled electricity transfer between states and the class label. The class label
    identifies the change of the price (UP or DOWN) in New South Wales relative to a moving average
    of the last 24 hours (and removes the impact of longer term price trends).

    Parameters:
        data_home (str): The directory where you wish to store the data.
        silent (bool): Whether to indicate download progress or not.

    Yields:
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.

    References:
        1. `SPLICE-2 Comparative Evaluation: Electricity Pricing <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.12.9405>`_
        2. `DataHub description <https://datahub.io/machine-learning/electricity#readme>`_

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

    The available raw features are: sender, recipients, date, subject, body.

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
    url = 'https://archive.org/download/trec07p/trec07p.tgz'

    # Download dataset if does not exist and get its path
    data_dir_path = download_dataset(name, url, data_home, archive_type='zip', silent=silent)

    return stream.iter_csv(f'{data_dir_path}/{name}.csv', target_name='y')


def fetch_credit_card(data_home=None, silent=True):
    """Credit card fraud dataset.

    The datasets contains transactions made by credit cards in September 2013 by european
    cardholders. This dataset presents transactions that occurred in two days, where we have 492
    frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class
    (frauds) account for 0.172% of all transactions.

    It contains only numerical input variables which are the result of a PCA transformation.
    Unfortunately, due to confidentiality issues, we cannot provide the original features and more
    background information about the data. Features V1, V2, ... V28 are the principal components
    obtained with PCA, the only features which have not been transformed with PCA are 'Time' and
    'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first
    transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be
    used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and
    it takes value 1 in case of fraud and 0 otherwise.

    Parameters:
        data_home (str): The directory where you wish to store the data.
        silent (bool): Whether to indicate download progress or not.

    Yields:
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.

    References:
        1. Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015
        2. Dal Pozzolo, Andrea; Caelen, Olivier; Le Borgne, Yann-Ael; Waterschoot, Serge; Bontempi, Gianluca. Learned lessons in credit card fraud detection from a practitioner perspective, Expert systems with applications,41,10,4915-4928,2014, Pergamon
        3. Dal Pozzolo, Andrea; Boracchi, Giacomo; Caelen, Olivier; Alippi, Cesare; Bontempi, Gianluca. Credit card fraud detection: a realistic modeling and a novel learning strategy, IEEE transactions on neural networks and learning systems,29,8,3784-3797,2018,IEEE
        4. Dal Pozzolo, Andrea Adaptive Machine learning for credit card fraud detection ULB MLG PhD thesis (supervised by G. Bontempi)
        5. Carcillo, Fabrizio; Dal Pozzolo, Andrea; Le Borgne, Yann-Ael; Caelen, Olivier; Mazzer, Yannis; Bontempi, Gianluca. Scarff: a scalable framework for streaming credit card fraud detection with Spark, Information fusion,41, 182-194,2018,Elsevier
        6. Carcillo, Fabrizio; Le Borgne, Yann-Ael; Caelen, Olivier; Bontempi, Gianluca. Streaming active learning strategies for real-life credit card fraud detection: assessment and visualization, International Journal of Data Science and Analytics, 5,4,285-300,2018,Springer International Publishing
        7. Bertrand Lebichot, Yann-Ael Le Borgne, Liyun He, Frederic Oble, Gianluca Bontempi Deep-Learning Domain Adaptation Techniques for Credit Cards Fraud Detection, INNSBDDL 2019: Recent Advances in Big Data and Deep Learning, pp 78-88, 2019
        8. Fabrizio Carcillo, Yann-Ael Le Borgne, Olivier Caelen, Frederic Oble, Gianluca Bontempi Combining Unsupervised and Supervised Learning in Credit Card Fraud Detection Information Sciences, 2019

    """

    name = 'creditcard'
    url = 'https://maxhalford.github.io/files/datasets/creditcardfraud.zip'

    data_dir_path = download_dataset(name, url, data_home, archive_type='zip', silent=silent)

    return stream.iter_csv(f'{data_dir_path}/{name}.csv', target_name='Class')


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
        os.path.join(os.path.dirname(__file__), 'airline-passengers.csv'),
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
        os.path.join(os.path.dirname(__file__), 'chick-weights.csv'),
        target_name='weight',
        converters={'time': int, 'weight': int, 'chick': int, 'diet': int}
    )
