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


def download_dataset(url, data_home, uncompress=True, verbose=True):
    """Downloads/decompresses a dataset locally if does not exist.

    Parameters:
        url (str): From where to download the dataset.
        data_home (str): The directory where you wish to store the data.
        uncompress (bool): Whether to uncompress the file or not.
        verbose (bool): Whether to indicate download progress or not.

    Returns:
        data_dir_path (str): Where the dataset is stored.

    """

    def _print(msg):
        if verbose:
            print(msg)

    name = os.path.basename(url)
    extension = '.'.join(name.split('.')[1:])
    data_home = get_data_home(data_home=data_home)
    path = os.path.join(data_home, f'{name}')
    archive_path = path
    if extension:
        path = path[:-(len(extension) + 1)]  # e.g. path/to/file.tar.gz becomes path/to/file

    # Download if necessary
    if not (os.path.exists(path) or os.path.exists(archive_path)):

        _print(f'Downloading {url}')
        with urllib.request.urlopen(url) as r, open(archive_path, 'wb') as f:
            shutil.copyfileobj(r, f)

    # If no uncompression is required then we're done
    if not uncompress:
        return archive_path

    # Uncompress if necessary
    if not os.path.exists(path):

        _print(f'Uncompressing into {path}')

        if extension.endswith('zip'):
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(path)

        elif extension.endswith(('gz', 'tar')):
            mode = 'r:' if extension.endswith('tar') else 'r:gz'
            print(archive_path, mode)
            tar = tarfile.open(archive_path, mode)
            tar.extractall(path)
            tar.close()

        else:
            raise RuntimeError(f'Unhandled extension type: {extension}')

        # Delete the archive file now that the dataset is available
        os.remove(archive_path)

    return path


class Dataset:

    def __init__(self, n_features, category):
        self.n_features = n_features
        self.category = category

    def __iter__(self):
        raise NotImplementedError

    def take(self, k):
        """Yields the k first (``x``, ``y``) pairs."""
        return itertools.islice(self, k)


class FileDataset(Dataset):

    def __init__(self, n_samples, n_features, category, **dl_params):
        super().__init__(n_features=n_features, category=category)
        self.n_samples = n_samples
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
