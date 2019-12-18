import itertools
import os
import shutil
import tarfile
import urllib
import zipfile


REG = 'Regression'
BINARY_CLF = 'Binary'
MULTI_CLF = 'Multiclass'


def get_data_home(data_home=None):
    """Return the path of the creme data directory.

    By default this will expand the relative path '~/creme_data'.

    """
    if data_home is None:
        data_home = os.environ.get('CREME_DATA', os.path.join('~', 'creme_data'))
        data_home = os.path.expanduser(data_home)
        if not os.path.exists(data_home):
            os.makedirs(data_home)
    return data_home


def download_dataset(name, url, data_home, archive_type=None, verbose=True):
    """Downloads/decompresses a dataset locally if does not exist.

    Parameters:
        name (str): Dataset filename or directory name.
        url (str): From where to download the dataset.
        data_home (str): The directory where you wish to store the data.
        archive_type (str): Dataset archive type extension name (e.g. 'zip'). Defaults to None.
        verbose (bool): Whether to indicate download progress or not.

    Returns:
        data_dir_path (str): Where the dataset is stored.

    """

    def _print(msg):
        if verbose:
            print(msg)

    data_home = get_data_home(data_home=data_home)
    data_dir_path = os.path.join(data_home, f'{name}')

    # Download if needed
    if not os.path.exists(data_dir_path):
        _print(f'Downloading {url}')

        if archive_type:
            data_dir_path = f'{data_dir_path}.{archive_type}'

        with urllib.request.urlopen(url) as r, open(data_dir_path, 'wb') as f:
            shutil.copyfileobj(r, f)

        # Uncompress if needed
        if archive_type:
            archive_path, data_dir_path = data_dir_path, data_dir_path[:-len(archive_type) - 1]
            _print(f'Uncompressing into {data_dir_path}')

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


class Dataset:

    def __init__(self, n_samples, n_features, category, **dl_params):
        self.n_samples = n_samples
        self.n_features = n_features
        self.category = category
        self.dl_params = dl_params

    def _stream_X_y(self, dir):
        raise NotImplementedError

    @property
    def _remote(self):
        """Whether or not the dataset needs downloading or not."""
        return 'url' in self.dl_params

    @property
    def _ready(self):
        """Whether or not the dataset is ready to be read."""
        if self._remote:
            return os.path.isdir(os.path.join(
                get_data_home(self.dl_params['data_home']),
                self.dl_params['name']
            ))
        return True

    def __iter__(self):
        if self._remote:
            data_dir_path = download_dataset(**self.dl_params)
        else:
            data_dir_path = os.path.dirname(__file__)
        yield from self._stream_X_y(data_dir_path)

    def take(self, k):
        """Returns the k first (``x``, ``y``) pairs."""
        return itertools.islice(self, k)
