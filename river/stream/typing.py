from __future__ import annotations

import os
import typing

if typing.TYPE_CHECKING:
    FilePath = str | os.PathLike[str]
    Compression = typing.Literal["infer", "gzip", "zip"]
    ResolvedCompression = typing.Literal["gzip", "zip"]
