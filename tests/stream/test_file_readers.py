from __future__ import annotations

import io
import typing
import zipfile

import pytest

from river import stream

if typing.TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable

    from river import base
    from tests.stream.conftest import CompressionCase, WriteFile

    Reader = Callable[..., base.typing.Stream]
    Rows = list[tuple[dict[str, typing.Any], typing.Any]]

READERS: dict[str, tuple[Reader, str, Rows]] = {
    "csv": (stream.iter_csv, "name\ncafé\n", [({"name": "café"}, None)]),
    "arff": (
        stream.iter_arff,
        "@relation test\n@attribute name {café}\n@data\ncafé\n",
        [({"name": "café"}, None)],
    ),
    "libsvm": (stream.iter_libsvm, "1 café:2\n", [({"café": 2.0}, 1.0)]),
}


@pytest.mark.parametrize(("read", "content", "expected"), READERS.values(), ids=READERS)
@pytest.mark.parametrize("encoding", [None, "utf-8", "latin-1"])
def test_file_readers_decode_paths(
    tmp_path: pathlib.Path,
    write_file: WriteFile,
    compression_case: CompressionCase,
    read: Reader,
    content: str,
    expected: Rows,
    encoding: str | None,
) -> None:
    path = write_file(
        tmp_path / f"data{compression_case.suffix}",
        content,
        compression_case.written_as,
        encoding=encoding or "utf-8",
    )
    kwargs = {} if encoding is None else {"encoding": encoding}

    assert list(read(path, compression=compression_case.read_as, **kwargs)) == expected


@pytest.mark.parametrize(("read", "content", "expected"), READERS.values(), ids=READERS)
def test_file_readers_leave_text_buffers_unchanged(
    read: Reader, content: str, expected: Rows
) -> None:
    buffer = io.StringIO(content)

    assert list(read(buffer, encoding="not-an-encoding")) == expected
    assert not buffer.closed


@pytest.mark.parametrize(("read", "content", "expected"), READERS.values(), ids=READERS)
@pytest.mark.parametrize("member_count", [0, 2])
def test_file_readers_reject_ambiguous_archives(
    tmp_path: pathlib.Path, read: Reader, content: str, expected: Rows, member_count: int
) -> None:
    path = tmp_path / "data.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for index in range(member_count):
            archive.writestr(f"data{index}", content)

    with pytest.raises(ValueError, match="exactly one file"):
        list(read(path))
