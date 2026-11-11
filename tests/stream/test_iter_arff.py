from __future__ import annotations

import gc
import io
import types
import typing

import pytest
import scipy.io.arff

from river import stream

if typing.TYPE_CHECKING:
    import pathlib

    from river.base.typing import FeatureName
    from tests.stream.conftest import CompressionCase, WriteFile

    Row = tuple[dict[FeatureName, typing.Any], typing.Any]

DENSE = """@relation test
@attribute a numeric
@attribute b numeric
@attribute c {x,y}
@data
1,2,x
3,?,y
"""

SPARSE = """@relation test
@attribute y0 {0,1}
@attribute y1 {0,1}
@attribute X0 numeric
@attribute X1 numeric
@data
{ 1 1,2 0.5 }
{ 0 1,3 0.25 }
"""

# The second row holds a `?`, so `b` is expected to be missing from its features.
UNTOUCHED: list[Row] = [({"a": 1.0, "b": 2.0, "c": "x"}, None), ({"a": 3.0, "c": "y"}, None)]
DENSE_CASES: dict[str, tuple[str | list[str] | None, list[Row]]] = {
    "target": ("c", [({"a": 1.0, "b": 2.0}, "x"), ({"a": 3.0}, "y")]),
    "no-target": (None, UNTOUCHED),
    "absent-target": ("nope", UNTOUCHED),
    "targets": (
        ["b", "c"],
        [({"a": 1.0}, {"b": 2.0, "c": "x"}), ({"a": 3.0}, {"b": None, "c": "y"})],
    ),
}


@pytest.mark.parametrize(("target", "expected"), DENSE_CASES.values(), ids=DENSE_CASES)
def test_numeric_attributes_are_cast_and_the_target_is_split_off(
    target: str | list[str] | None, expected: list[Row]
) -> None:
    assert list(stream.iter_arff(io.StringIO(DENSE), target=target)) == expected


def test_sparse_rows_only_carry_the_features_they_list() -> None:
    dataset = stream.iter_arff(io.StringIO(SPARSE), target=["y0", "y1"], sparse=True)

    assert list(dataset) == [
        ({"X0": 0.5}, {"y0": "0", "y1": "1"}),
        ({"X1": 0.25}, {"y0": "1", "y1": "0"}),
    ]


def test_a_dataset_is_read_from_a_path(
    tmp_path: pathlib.Path, write_file: WriteFile, compression_case: CompressionCase
) -> None:
    path = write_file(
        tmp_path / f"data.arff{compression_case.suffix}", DENSE, compression_case.written_as
    )

    dataset = stream.iter_arff(path, target="c", compression=compression_case.read_as)
    assert list(dataset) == DENSE_CASES["target"][1]


def test_a_file_the_reader_opened_is_closed(
    tmp_path: pathlib.Path,
    write_file: WriteFile,
    compression_case: CompressionCase,
    opened_files: list[typing.TextIO],
) -> None:
    path = write_file(
        tmp_path / f"data.arff{compression_case.suffix}", DENSE, compression_case.written_as
    )

    _ = list(stream.iter_arff(path, target="c", compression=compression_case.read_as))

    assert len(opened_files) == 1
    assert opened_files[0].closed


def test_a_buffer_passed_in_by_the_caller_is_left_open() -> None:
    buffer = io.StringIO(DENSE)

    _ = list(stream.iter_arff(buffer, target="c"))

    assert not buffer.closed


def test_an_unparsable_header_is_reported_as_a_parse_error() -> None:
    dataset = stream.iter_arff(io.StringIO("@relation test\n@attribute a bogus\n@data\n1\n"))

    with pytest.raises(scipy.io.arff.ParseArffError):
        next(dataset)


SPARSE_HEADER = """@relation test
@attribute a numeric
@attribute score numeric
@attribute label {negative,positive}
@data
"""

STRING_SPARSE_HEADER = """@relation test
@attribute a numeric
@attribute label string
@data
"""


@pytest.mark.parametrize(
    "row",
    [
        "{0 1.5,1 2.5,2 positive}",
        "{0 1.5, 1 2.5, 2 positive}",
        "  { 0 1.5, 1 2.5, 2 positive }  ",
        "{0\t1.5,\t1\t2.5,\t2\tpositive}",
        "{2 positive ,0 1.5,1 2.5}",
    ],
)
def test_sparse_values_are_cast_with_whitespace_between_entries(row: str) -> None:
    dataset = stream.iter_arff(io.StringIO(SPARSE_HEADER + row + "\n"), target="label", sparse=True)

    assert list(dataset) == [({"a": 1.5, "score": 2.5}, "positive")]


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ("{0 1.5,1 'class A'}", "class A"),
        ("{0 1.5,1 'class,A'}", "class,A"),
        ('{0 1.5,1 "class,A"}', "class,A"),
        ("{1 'class,A'   ,0 1.5}", "class,A"),
        ("{0 1.5,1 '?'}", "?"),
        (r"{0 1.5,1 '\''}", "'"),
        (r"{0 1.5,1 '\\'}", "\\"),
        (r"{0 1.5,1 '\\\''}", "\\'"),
        (r"{0 1.5,1 '\n'}", "\n"),
        (r"{0 1.5,1 '\r'}", "\r"),
        (r"{0 1.5,1 '\t'}", "\t"),
        (r"{0 1.5,1 '\b'}", "\b"),
        (r"{0 1.5,1 '\f'}", "\f"),
        (r"{0 1.5,1 '\0'}", "\0"),
        (r"{0 1.5,1 '\01'}", "\01"),
        (r"{0 1.5,1 '\011'}", "\t"),
        (r"{0 1.5,1 '\u123a'}", "ሺ"),
        (r"{0 1.5,1 '\%'}", "%"),
    ],
)
def test_sparse_quoted_values_are_unquoted(row: str, expected: str) -> None:
    dataset = stream.iter_arff(
        io.StringIO(STRING_SPARSE_HEADER + row + "\n"), target="label", sparse=True
    )

    assert list(dataset) == [({"a": 1.5}, expected)]


@pytest.mark.parametrize(
    "row",
    [
        r"{1 \'A' }",
        r"{1 \'A }",
        "{1 'A }",
        '{1 "A }',
        '{1 A" }',
        "{1 A' }",
        r"{1 'A\' }",
        r'{1 "A\" }',
        r"{1 '\x00'}",
        r"{1 '\u1'}",
        r"{1 '\uzzzz'}",
        r"{1 '\N'}",
        r"{1 '\T'}",
        "{1 A B}",
        "{1 5 6}",
        "{1 '5' '6'}",
        "0 {1 A }",
        "0, {1 A }",
        "{1 A } 0",
        "{1 A }, 0",
        "{1 A",
        "1 A}",
    ],
)
def test_invalid_sparse_layouts_raise(row: str) -> None:
    dataset = stream.iter_arff(io.StringIO(STRING_SPARSE_HEADER + row + "\n"), sparse=True)

    with pytest.raises(ValueError):
        next(dataset)


@pytest.mark.parametrize("row", ["{0 nope}", "{2 unknown}", "{3 1}", "{-1 1}"])
def test_invalid_sparse_values_raise(row: str) -> None:
    dataset = stream.iter_arff(io.StringIO(SPARSE_HEADER + row + "\n"), sparse=True)

    with pytest.raises(ValueError):
        next(dataset)


@pytest.mark.parametrize(
    ("row", "expected_features"), [("{}", {}), ("{ }", {}), ("{0 1.5}", {"a": 1.5})]
)
@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("score", 0.0),
        ("label", "negative"),
        (["score", "label"], {"score": 0.0, "label": "negative"}),
    ],
)
def test_sparse_omitted_targets_use_the_attribute_default(
    row: str, expected_features: dict[str, float], target: str | list[str], expected: typing.Any
) -> None:
    dataset = stream.iter_arff(io.StringIO(SPARSE_HEADER + row + "\n"), target=target, sparse=True)

    features, actual = next(dataset)
    assert actual == expected
    assert features == expected_features


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("score", None),
        ("label", None),
        (["score", "label"], {"score": None, "label": None}),
    ],
)
def test_explicitly_missing_targets_are_none(
    sparse: bool, target: str | list[str], expected: typing.Any
) -> None:
    row = "{0 1.5,1 ?,2 ?}" if sparse else "1.5,?,?"
    dataset = stream.iter_arff(
        io.StringIO(SPARSE_HEADER + row + "\n"), target=target, sparse=sparse
    )

    assert list(dataset) == [({"a": 1.5}, expected)]


def test_sparse_missing_features_are_not_reported_as_values() -> None:
    dataset = stream.iter_arff(
        io.StringIO(SPARSE_HEADER + "{0 ?,1 2.5,2 positive}\n"), target="label", sparse=True
    )

    assert list(dataset) == [({"score": 2.5}, "positive")]


def test_sparse_omitted_features_remain_omitted() -> None:
    assert list(stream.iter_arff(io.StringIO(SPARSE_HEADER + "{}\n"), sparse=True)) == [({}, None)]


def test_sparse_blank_and_comment_lines_do_not_yield_observations() -> None:
    dataset = stream.iter_arff(
        io.StringIO(SPARSE_HEADER + "\n  \n% comment\n   % comment\n{0 1.5}\n"), sparse=True
    )

    assert list(dataset) == [({"a": 1.5}, None)]


@pytest.mark.parametrize(
    ("row", "expected_features", "expected_targets"),
    [
        ("{1 ? ,0 1.5}", {"a": 1.5}, {"score": None, "label": "negative"}),
        ("{2 ? ,0 1.5}", {"a": 1.5}, {"score": 0.0, "label": None}),
        ("{0 ? ,1 2.5,2 positive}", {}, {"score": 2.5, "label": "positive"}),
    ],
)
def test_sparse_missing_values_with_padding(
    row: str, expected_features: dict[str, float], expected_targets: dict[str, typing.Any]
) -> None:
    dataset = stream.iter_arff(
        io.StringIO(SPARSE_HEADER + row + "\n"), target=["score", "label"], sparse=True
    )

    assert list(dataset) == [(expected_features, expected_targets)]


@pytest.mark.parametrize("termination", ["close", "finalize"])
def test_interrupted_owned_files_are_closed(
    tmp_path: pathlib.Path,
    write_file: WriteFile,
    compression_case: CompressionCase,
    opened_files: list[typing.TextIO],
    termination: str,
) -> None:
    path = write_file(
        tmp_path / f"data.arff{compression_case.suffix}",
        DENSE,
        compression_case.written_as,
    )
    dataset = stream.iter_arff(path, compression=compression_case.read_as)
    _ = next(dataset)
    assert len(opened_files) == 1
    assert not opened_files[0].closed

    if termination == "close":
        assert isinstance(dataset, types.GeneratorType)
        dataset.close()
    else:
        del dataset
        _ = gc.collect()

    assert opened_files[0].closed


@pytest.mark.parametrize("termination", ["close", "finalize"])
def test_interrupted_caller_buffers_are_left_open(termination: str) -> None:
    buffer = io.StringIO(DENSE)
    dataset = stream.iter_arff(buffer)
    _ = next(dataset)

    if termination == "close":
        assert isinstance(dataset, types.GeneratorType)
        dataset.close()
    else:
        del dataset
        _ = gc.collect()

    assert not buffer.closed


@pytest.mark.parametrize(
    ("content", "sparse", "error"),
    [
        ("@relation test\n@attribute a bogus\n@data\n1\n", False, scipy.io.arff.ParseArffError),
        (DENSE + "bad,2,x\n", False, ValueError),
        (SPARSE + "{ 2 bad }\n", True, ValueError),
        (SPARSE + "bad row\n", True, ValueError),
    ],
)
def test_owned_files_are_closed_after_errors(
    tmp_path: pathlib.Path,
    write_file: WriteFile,
    compression_case: CompressionCase,
    opened_files: list[typing.TextIO],
    content: str,
    sparse: bool,
    error: type[Exception],
) -> None:
    path = write_file(
        tmp_path / f"data.arff{compression_case.suffix}",
        content,
        compression_case.written_as,
    )
    dataset = stream.iter_arff(path, compression=compression_case.read_as, sparse=sparse)

    with pytest.raises(error):
        list(dataset)

    assert len(opened_files) == 1
    assert opened_files[0].closed


@pytest.mark.parametrize(
    ("content", "sparse", "error"),
    [
        ("@relation test\n@attribute a bogus\n@data\n1\n", False, scipy.io.arff.ParseArffError),
        (DENSE + "bad,2,x\n", False, ValueError),
        (SPARSE + "{ 2 bad }\n", True, ValueError),
        (SPARSE + "bad row\n", True, ValueError),
    ],
)
def test_caller_buffers_are_left_open_after_errors(
    content: str,
    sparse: bool,
    error: type[Exception],
) -> None:
    buffer = io.StringIO(content)

    with pytest.raises(error):
        list(stream.iter_arff(buffer, sparse=sparse))

    assert not buffer.closed
