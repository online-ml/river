from __future__ import annotations

import functools
import gzip
import io
import os
import zipfile


def open_filepath(filepath_or_buffer, compression):
    # Determine the compression from the file extension if "infer" has been specified
    if compression == "infer":
        _, ext = os.path.splitext(filepath_or_buffer)
        compression = {".gz": "gzip", ".zip": "zip"}.get(ext)

    def open_zipfile(path):
        with zipfile.ZipFile(path, "r") as zf:
            f = zf.open(zf.namelist()[0], "r")
            f = io.TextIOWrapper(f)
            return f

    # Determine the file opening method from the compression
    open_func = {
        None: open,
        "gzip": functools.partial(gzip.open, mode="rt"),
        "zip": open_zipfile,
    }[compression]

    # Open the file using the opening method
    return open_func(filepath_or_buffer)
