from __future__ import annotations

import io
import typing

import pytest

from river import stream

if typing.TYPE_CHECKING:
    import pathlib

    from river.base.typing import FeatureName
    from tests.stream.conftest import CompressionCase, WriteFile

CONTENT = "+1 x:-134.26 y:0.2563\n1 x:-12 z:0.3\n-1 y:.25\n"
FEATURES: list[dict[FeatureName, float]] = [
    {"x": -134.26, "y": 0.2563},
    {"x": -12.0, "z": 0.3},
    {"y": 0.25},
]

# `1.0 == 1`, so the labels are compared by type as well as by value.
TARGET_TYPE_CASES: dict[str, tuple[type[typing.Any], list[typing.Any]]] = {
    "float": (float, [1.0, 1.0, -1.0]),
    "int": (int, [1, 1, -1]),
    "str": (str, ["+1", "1", "-1"]),
}


@pytest.mark.parametrize(
    ("target_type", "targets"), TARGET_TYPE_CASES.values(), ids=TARGET_TYPE_CASES
)
def test_the_label_is_cast_to_the_target_type(
    target_type: type[typing.Any], targets: list[typing.Any]
) -> None:
    rows = list(stream.iter_libsvm(io.StringIO(CONTENT), target_type=target_type))

    assert rows == list(zip(FEATURES, targets))
    assert [type(y) for _, y in rows] == [target_type] * len(targets)


def test_the_label_is_a_float_by_default() -> None:
    rows = list(stream.iter_libsvm(io.StringIO(CONTENT)))

    assert rows == list(zip(FEATURES, [1.0, 1.0, -1.0]))
    assert [type(y) for _, y in rows] == [float, float, float]


def test_trailing_comments_are_ignored() -> None:
    dataset = stream.iter_libsvm(io.StringIO("+1 x:1#a comment\n"), target_type=int)

    assert list(dataset) == [({"x": 1.0}, 1)]


def test_a_dataset_is_read_from_a_path(
    tmp_path: pathlib.Path, write_file: WriteFile, compression_case: CompressionCase
) -> None:
    path = write_file(
        tmp_path / f"data.svm{compression_case.suffix}", CONTENT, compression_case.written_as
    )

    dataset = stream.iter_libsvm(path, target_type=int, compression=compression_case.read_as)
    assert list(dataset) == list(zip(FEATURES, [1, 1, -1]))


def test_a_file_the_reader_opened_is_closed(
    tmp_path: pathlib.Path,
    write_file: WriteFile,
    compression_case: CompressionCase,
    opened_files: list[typing.TextIO],
) -> None:
    path = write_file(
        tmp_path / f"data.svm{compression_case.suffix}", CONTENT, compression_case.written_as
    )

    _ = list(stream.iter_libsvm(path, compression=compression_case.read_as))

    assert len(opened_files) == 1
    assert opened_files[0].closed


def test_a_buffer_passed_in_by_the_caller_is_left_open() -> None:
    buffer = io.StringIO(CONTENT)

    _ = list(stream.iter_libsvm(buffer))

    assert not buffer.closed
