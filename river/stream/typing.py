from __future__ import annotations

import os
import typing

FilePath = str | os.PathLike[str]
"""Anything the built-in open() accepts as a file location."""
ResolvedCompression = typing.Literal["gzip", "zip"]
"""A decompression method that maps to a concrete reader."""
Compression = ResolvedCompression | typing.Literal["infer"]
"""A caller-facing decompression method, "infer" being resolved from the file extension."""
