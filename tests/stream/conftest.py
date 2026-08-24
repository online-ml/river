from __future__ import annotations

import gzip
import typing
import zipfile

import pytest

from river.stream import utils

if typing.TYPE_CHECKING:
    import pathlib

    from river.stream.typing import Compression, FilePath

    class WriteFile(typing.Protocol):
        """Writes `content` to `path`, compressed as `compression` ("infer" uses the extension)."""

        def __call__(
            self, path: pathlib.Path, content: str, compression: Compression | None = "infer"
        ) -> pathlib.Path: ...


EXT_TO_FORMAT = {".gz": "gzip", ".zip": "zip"}


class CompressionCase(typing.NamedTuple):
    """An extension to write a file under, the format to write it in, and how to read it back."""

    suffix: str
    written_as: Compression | None
    read_as: Compression | None


# The last two cases give the file an extension that contradicts its content, so that they only
# pass if the explicit `compression` argument wins over the inference.
COMPRESSION_CASES = {
    "plain": CompressionCase("", "infer", "infer"),
    "gzip-inferred": CompressionCase(".gz", "infer", "infer"),
    "zip-inferred": CompressionCase(".zip", "infer", "infer"),
    "gzip-explicit": CompressionCase("", "gzip", "gzip"),
    "none-explicit": CompressionCase(".gz", None, None),
}


@pytest.fixture(params=COMPRESSION_CASES.values(), ids=list(COMPRESSION_CASES))
def compression_case(request: pytest.FixtureRequest) -> CompressionCase:
    return typing.cast("CompressionCase", request.param)


@pytest.fixture
def write_file() -> WriteFile:
    def write(
        path: pathlib.Path, content: str, compression: Compression | None = "infer"
    ) -> pathlib.Path:
        match EXT_TO_FORMAT.get(path.suffix) if compression == "infer" else compression:
            case "gzip":
                with gzip.open(path, mode="wt") as f:
                    _ = f.write(content)
            case "zip":
                with zipfile.ZipFile(path, mode="w") as archive:
                    archive.writestr(path.stem, content)
            case _:
                _ = path.write_text(content)
        return path

    return write


@pytest.fixture
def opened_files(monkeypatch: pytest.MonkeyPatch) -> list[typing.TextIO]:
    """Records the files the readers open themselves, so tests can check they close them."""
    handles: list[typing.TextIO] = []
    open_filepath = utils.open_filepath

    def spy(filepath: FilePath, compression: Compression | None) -> typing.TextIO:
        handle = open_filepath(filepath, compression)
        handles.append(handle)
        return handle

    monkeypatch.setattr(utils, "open_filepath", spy)
    return handles
