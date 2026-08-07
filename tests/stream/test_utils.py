from __future__ import annotations

import typing

import pytest

from river.stream import utils

if typing.TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable

    from river.stream.typing import Compression
    from tests.stream.conftest import CompressionCase, WriteFile

    ReadBack = Callable[[typing.TextIO], typing.Any]

CONTENT = "name,rating\na,9.5\nb,9.4\n"
LINES = ["name,rating\n", "a,9.5\n", "b,9.4\n"]

READ_BACK_CASES: dict[str, tuple[ReadBack, typing.Any]] = {
    "whole": (lambda f: f.read(), CONTENT),
    "line-by-line": (lambda f: list(f), LINES),
}


@pytest.mark.parametrize(("read_back", "expected"), READ_BACK_CASES.values(), ids=READ_BACK_CASES)
@pytest.mark.parametrize("as_str", [False, True], ids=["pathlike", "str"])
def test_the_content_is_read_back_as_text(
    tmp_path: pathlib.Path,
    write_file: WriteFile,
    compression_case: CompressionCase,
    as_str: bool,
    read_back: ReadBack,
    expected: typing.Any,
) -> None:
    path = write_file(
        tmp_path / f"data.csv{compression_case.suffix}", CONTENT, compression_case.written_as
    )

    with utils.open_filepath(str(path) if as_str else path, compression_case.read_as) as f:
        assert read_back(f) == expected


def test_an_unknown_extension_is_read_as_is(tmp_path: pathlib.Path, write_file: WriteFile) -> None:
    with utils.open_filepath(write_file(tmp_path / "data.unknown", CONTENT), "infer") as f:
        assert f.read() == CONTENT


@pytest.mark.parametrize("compression", ["infer", "gzip", "zip", None])
def test_a_missing_file_raises(tmp_path: pathlib.Path, compression: Compression | None) -> None:
    with pytest.raises(FileNotFoundError):
        utils.open_filepath(tmp_path / "nope.csv", compression)


def test_a_zip_member_outlives_the_archive_object(
    tmp_path: pathlib.Path, write_file: WriteFile
) -> None:
    """`open_filepath` returns a member of a `ZipFile` it has already closed."""
    handle = utils.open_filepath(write_file(tmp_path / "data.csv.zip", CONTENT), "infer")
    try:
        assert handle.read() == CONTENT
    finally:
        handle.close()
    assert handle.closed
