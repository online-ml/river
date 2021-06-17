import glob
import inspect
import os
import pickle
import platform

from river import utils


class Cache:
    """Utility for caching iterables.

    This can be used to save a stream of data to the disk in order to iterate over it faster the
    following time. This can save time depending on the nature of stream. The more processing
    happens in a stream, the more time will be saved. Even in the case where no processing is done
    apart from reading the data, the cache will save some time because it is using the pickle
    binary protocol. It can thus improve the speed in common cases such as reading from a CSV file.

    Parameters
    ----------
    directory
        The path where to store the pickled data streams. If not provided, then it will be
        automatically inferred whenever possible, if not an exception will be raised.

    Attributes
    ----------
    keys : set
        The set of keys that are being cached.

    Examples
    --------

    >>> import time
    >>> from river import datasets
    >>> from river import stream

    >>> dataset = datasets.Phishing()
    >>> cache = stream.Cache()

    The cache can be used by wrapping it around an iterable. Because this is the first time
    are iterating over the data, nothing is cached.

    >>> tic = time.time()
    >>> for x, y in cache(dataset, key='phishing'):
    ...     pass
    >>> toc = time.time()
    >>> print(toc - tic)  # doctest: +SKIP
    0.012813

    If we do the same thing again, we can see the loop is now faster.

    >>> tic = time.time()
    >>> for x, y in cache(dataset, key='phishing'):
    ...     pass
    >>> toc = time.time()
    >>> print(toc - tic)  # doctest: +SKIP
    0.001927

    We can see an overview of the cache. The first line indicates the location of the
    cache.

    >>> cache  # doctest: +SKIP
    /tmp
    phishing - 125.2KiB

    Finally, we can clear the stream from the cache.

    >>> cache.clear('phishing')
    >>> cache  # doctest: +SKIP
    /tmp

    There is also a `clear_all` method to remove all the items in the cache.

    >>> cache.clear_all()

    """

    def __init__(self, directory=None):

        # Guess the directory from the system
        system = platform.system()
        if directory is None:
            directory = {"Linux": "/tmp", "Darwin": "/tmp", "Windows": "C:\\TEMP"}.get(
                system
            )

        if directory is None:
            raise ValueError(
                f"There is no default directory defined for {system} systems, "
                "please provide one manually"
            )

        self.directory = directory
        self.keys = set()

        # Check if there is anything already in the cache
        for f in glob.glob(os.path.join(self.directory, "*.river_cache.pkl")):
            key = os.path.basename(f).split(".")[0]
            self.keys.add(key)

    def _get_path(self, key):
        return os.path.join(self.directory, f"{key}.river_cache.pkl")

    def __call__(self, stream, key=None):

        # Try to guess a key from the stream object
        if key is None:
            if inspect.isfunction(stream):
                key = stream.__name__

        if key is None:
            raise ValueError(
                "No default key could be guessed for the given stream, "
                "please provide one"
            )

        path = self._get_path(key)

        if os.path.exists(path):
            yield from self[key]
            return

        with open(path, "wb") as f:
            pickler = pickle.Pickler(f)
            for el in stream:
                pickler.dump(el)
                yield el
            self.keys.add(key)

    def __getitem__(self, key):
        """Iterates over the stream associated with the given key."""
        with open(self._get_path(key), "rb") as f:
            unpickler = pickle.Unpickler(f)
            while f.peek(1):
                yield unpickler.load()

    def clear(self, key: str):
        """Delete the cached stream associated with the given key.

        Parameters
        ----------
        key

        """
        os.remove(self._get_path(key))
        self.keys.remove(key)

    def clear_all(self):
        """Delete all the cached streams."""
        for key in list(self.keys):
            os.remove(self._get_path(key))
            self.keys.remove(key)

    def __repr__(self):
        return "\n".join(
            [self.directory]
            + [
                f"{key} - {utils.pretty.humanize_bytes(os.path.getsize(self._get_path(key)))}"
                for key in self.keys
            ]
        )
