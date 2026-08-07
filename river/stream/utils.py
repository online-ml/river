from __future__ import annotations

import gzip
import io
import os
import typing
import zipfile

if typing.TYPE_CHECKING:
    from river.stream.typing import Compression, FilePath, ResolvedCompression

_EXT_TO_COMPRESSION: dict[str, ResolvedCompression] = {".gz": "gzip", ".zip": "zip"}
"""File extensions that "infer" knows how to turn into a decompression method."""


def open_filepath(filepath: FilePath, compression: Compression | None) -> typing.TextIO:
    """Open a possibly compressed file in text mode.

    Parameters
    ----------
    filepath
        Location of the file to open.
    compression
        Decompression method to use. `None` opens the file as-is, whereas "infer" picks the method
        from the file extension, and opens the file as-is for unknown extensions.

    Returns
    -------
    A text-mode file object, which the caller is responsible for closing.

    """
    resolved = (
        _EXT_TO_COMPRESSION.get(os.path.splitext(filepath)[1])
        if compression == "infer"
        else compression
    )

    match resolved:
        case None:
            return open(filepath)
        case "gzip":
            return gzip.open(filepath, mode="rt")
        case "zip":
            return _open_zipfile(filepath)
        case _:
            typing.assert_never(resolved)


def _open_zipfile(path: FilePath) -> typing.TextIO:
    with zipfile.ZipFile(path, "r") as zf:
        # Closing the archive does not close the member handles it handed out, so the wrapper
        # stays readable after this block.
        return io.TextIOWrapper(zf.open(zf.namelist()[0], "r"))
