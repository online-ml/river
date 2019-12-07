from .. import stream

from . import base


class MovieLens100k(base.Dataset):
    """MovieLens 100K dataset.

    MovieLens datasets were collected by the GroupLens Research Project at the University of
    Minnesota. This dataset consists of 100,000 ratings (1-5) from 943 users on 1682 movies. Each
    user has rated at least 20 movies. User and movie informations are provided. The data was
    collected through the MovieLens web site (movielens.umn.edu) during the seven-month period from
    September 19th, 1997 through April 22nd, 1998.

    Parameters:
        data_home (str): The directory where you wish to store the data.
        verbose (bool): Whether to indicate download progress or not.

    Yields:
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.

    References:
        1. `The MovieLens Datasets: History and Context <http://dx.doi.org/10.1145/2827872>`_

    """

    def __init__(self, data_home=None, verbose=False):
        super().__init__(
            n_samples=100_000,
            n_features=10,
            category=base.REG,
            name='ml-100k',
            url='https://maxhalford.github.io/files/datasets/ml-100k.zip',
            data_home=data_home,
            archive_type='zip',
            verbose=verbose
        )

    def _stream_X_y(self, directory):
        return stream.iter_csv(
            f'{directory}/ml_100k.csv',
            target_name='rating',
            converters={
                'user': str,
                'item': str,
                'timestamp': int,
                'title': str,
                'release_date': int,
                'genres': str,
                'age': float,
                'gender': float,
                'occupation': str,
                'zip_code': str,
                'rating': float
            }
        )
