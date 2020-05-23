import abc
import itertools
import pathlib
import os
import shutil
import tarfile
from urllib import request
import zipfile

from creme import utils


REG = 'Regression'
BINARY_CLF = 'Binary classification'
MULTI_CLF = 'Multi-class classification'
MO_BINARY_CLF = 'Multi-output binary classification'


def get_data_home():
    """Return the location where remote datasets are to be stored.

    """

    data_home = os.environ.get('CREME_DATA', os.path.join('~', 'creme_data'))
    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    return data_home


class Dataset(abc.ABC):
    """Base class for all datasets.

    All datasets inherit from this class, be they stored in a file or generated on the fly.

    """

    def __init__(self, task, n_features, n_samples=None, n_classes=None, sparse=False):
        self.task = task
        self.n_features = n_features
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.sparse = sparse

    @abc.abstractmethod
    def __iter__(self):
        raise NotImplementedError

    def take(self, k: int):
        """Iterate over the k samples."""
        return itertools.islice(self, k)

    @property
    def _repr_title(self):
        return f'{self.__class__.__name__} dataset'

    @property
    def _repr_content(self):
        """The items that are displayed in the __repr__ method.

        This property can be overriden in order to modify the output of the __repr__ method.

        """

        content = {}
        content['Task'] = self.task
        if self.n_samples:
            content['Number of samples'] = f'{self.n_samples:,}'
        if self.n_features:
            content['Number of features'] = f'{self.n_features:,}'
        if self.n_classes:
            content['Number of classes'] = f'{self.n_classes:,}'
        content['Sparse'] = str(self.sparse)

        return content

    def __repr__(self):

        l_len = max(map(len, self._repr_content.keys()))
        r_len = max(map(len, self._repr_content.values()))

        return (
            f'{self._repr_title}\n\n' +
            '\n'.join(
                k.rjust(l_len) + '  ' + v.ljust(r_len)
                for k, v in self._repr_content.items()
            )
        )


class SyntheticDataset(Dataset):

    @property
    def _repr_title(self):
        return f'{self.__class__.__name__} synthetic dataset'



class FileDataset(Dataset):
    """Base class for datasets that are stored in a local file.

    Small datasets that are part of the creme package inherit from this class.

    """

    def __init__(self, filename, **desc):
        super().__init__(**desc)
        self.filename = filename

    @property
    def path(self):
        return pathlib.Path(__file__).parent.joinpath(self.filename)

    @property
    def _repr_content(self):
        content = super()._repr_content
        content['Path'] = str(self.path)
        return content


class RemoteDataset(FileDataset):
    """Base class for datasets that are stored in a remote file.

    Medium and large datasets that are not part of the creme package inherit from this class.

    The filename doesn't have to be provided if unpack is False. Indeed in the latter case the
    filename will be inferred from the URL.

    """

    def __init__(self, url, size, filename=None, unpack=True, **desc):

        if filename is None:
            filename = os.path.basename(url)

        super().__init__(filename=filename, **desc)
        self.url = url
        self.size = size
        self.unpack = unpack

    @property
    def path(self):
        return pathlib.Path(get_data_home(), self.__class__.__name__, self.filename)

    def download(self, verbose=True):

        # Determine where to download the archive
        directory = self.path.parent
        directory.mkdir(parents=True, exist_ok=True)
        archive_path = directory.joinpath(os.path.basename(self.url))

        with request.urlopen(self.url) as r:

            # Notify the user
            if verbose:
                meta = r.info()
                try:
                    n_bytes = int(meta['Content-Length'])
                    msg = f'Downloading {self.url} ({utils.pretty.humanize_bytes(n_bytes)})'
                except KeyError:
                    msg = f'Downloading {self.url}'
                print(msg)

            # Now dump the contents of the requests
            with open(archive_path, 'wb') as f:
                shutil.copyfileobj(r, f)

        if not self.unpack:
            return

        if verbose:
            print(f'Uncompressing into {directory}')

        if archive_path.suffix.endswith('zip'):
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(directory)

        elif archive_path.suffix.endswith(('gz', 'tar')):
            mode = 'r:' if archive_path.suffix.endswith('tar') else 'r:gz'
            tar = tarfile.open(archive_path, mode)
            tar.extractall(directory)
            tar.close()

        else:
            raise RuntimeError(f'Unhandled extension type: {archive_path.suffix}')

        # Delete the archive file now that it has been uncompressed
        archive_path.unlink()

    @abc.abstractmethod
    def _iter(self):
        pass

    @property
    def is_downloaded(self):
        """Indicate whether or the data has been correctly downloaded."""
        if self.path.exists():

            if self.path.is_file():
                return self.path.stat().st_size == self.size
            return sum(f.stat().st_size for f in self.path.glob('**/*') if f.is_file())

        return False

    def __iter__(self):
        if not self.is_downloaded:
            self.download(verbose=True)
        if not self.is_downloaded:
            raise RuntimeError('Something went wrong during the download')
        yield from self._iter()

    @property
    def _repr_content(self):
        content = super()._repr_content
        content['URL'] = self.url
        content['Size'] = utils.pretty.humanize_bytes(self.size)
        content['Downloaded'] = str(self.is_downloaded)
        return content
