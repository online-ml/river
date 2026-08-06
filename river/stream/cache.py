from __future__ import annotations

import glob
import inspect
import os
import pickle
import platform
import typing

from river import utils

if typing.TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from river.stream.typing import FilePath

T = typing.TypeVar("T")

CACHE_SUFFIX = ".river_cache.pkl"
"""Extension of a cached stream, which is also what identifies one in a shared directory."""

_SYSTEM_TO_DIR: dict[str, str] = {"Linux": "/tmp", "Darwin": "/tmp", "Windows": "C:\\TEMP"}
"""Where to cache streams on each `platform.system()`."""


class Cache:
    """Utility for caching iterables.

    This can be used to save a stream of data to the disk in order to iterate over it faster the
    following time. This can save time depending on the nature of the stream. The more processing
    happens in a stream, the more time will be saved. Even in the case where no processing is done
    apart from reading the data, the cache will save some time because it is using the pickle
    binary protocol. It can thus improve the speed in common cases such as reading from a CSV file.

    Parameters
    ----------
    directory
        The existing directory where to store the pickled data streams. If not provided, then a
        temporary directory is picked for the current system, and an exception is raised if none
        is known.

    Attributes
    ----------
    keys : set[str]
        The set of keys that are being cached.

    Examples
    --------

    >>> import time
    >>> from river import datasets
    >>> from river import stream

    >>> dataset = datasets.Phishing()
    >>> cache = stream.Cache()

    The cache can be used by wrapping it around an iterable. Because this is the first time we
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

    def __init__(self, directory: FilePath | None = None) -> None:
        # Guess the directory from the system
        if directory is None:
            system = platform.system()
            if (directory := _SYSTEM_TO_DIR.get(system)) is None:
                raise ValueError(
                    f"There is no default directory defined for {system} systems, "
                    "please provide one manually"
                )

        # os.fspath keeps the rest of the class working on plain strings, which os.path.join
        # accepts either way but "\n".join in __repr__ does not.
        self.directory: str = os.fspath(directory)

        # Pick up whatever a previous instance left in the directory
        self.keys: set[str] = {
            os.path.basename(path).removesuffix(CACHE_SUFFIX)
            for path in glob.glob(os.path.join(self.directory, f"*{CACHE_SUFFIX}"))
        }

    def _get_path(self, key: str) -> str:
        return os.path.join(self.directory, f"{key}{CACHE_SUFFIX}")

    def __call__(self, stream: Iterable[T], key: str | None = None) -> Iterator[T]:
        """Iterates over a stream, caching it along the way if it is not cached yet.

        Parameters
        ----------
        stream
            The iterable to cache. It is only consumed on the first pass; later passes read the
            elements back from the disk and ignore it entirely.
        key
            The name to cache the stream under. Two different streams therefore need two
            different keys, as a key that is already cached is what makes a pass a cache hit.

        Raises
        ------
        ValueError
            If no `key` is given and none can be guessed from `stream`.

        """
        # Try to guess a key from the stream object
        if key is None:
            if inspect.isfunction(stream):
                key = stream.__name__

        if key is None:
            msg = "No default key could be guessed for the given stream, please provide one"
            raise ValueError(msg)

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

    def __getitem__(self, key: str) -> Iterator[typing.Any]:
        """Iterates over the stream associated with the given key.

        Parameters
        ----------
        key
            The name the stream was cached under.

        """
        with open(self._get_path(key), "rb") as f:
            unpickler = pickle.Unpickler(f)
            while f.peek(1):
                yield unpickler.load()

    def clear(self, key: str) -> None:
        """Delete the cached stream associated with the given key.

        Parameters
        ----------
        key
            The name the stream was cached under.

        """
        os.remove(self._get_path(key))
        self.keys.remove(key)

    def clear_all(self) -> None:
        """Delete all the cached streams."""
        for key in list(self.keys):
            os.remove(self._get_path(key))
            self.keys.remove(key)

    def __repr__(self) -> str:
        parts = (
            f"{key} - {utils.pretty.humanize_bytes(os.path.getsize(self._get_path(key)))}"
            for key in self.keys
        )
        return "\n".join((self.directory, *parts))
