from __future__ import annotations

import io
import typing

import pytest

from river import stream

HEADER = """@relation test
@attribute amount numeric
@attribute text string
@attribute label {negative,positive}
@data
"""


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("plain", "plain"),
        ("'hello, world'", "hello, world"),
        ('"hello, world"', "hello, world"),
        ("'100%, complete'", "100%, complete"),
        ("' leading and trailing '", " leading and trailing "),
        ("'?'", "?"),
        ('"?"', "?"),
        ("''", ""),
        ('""', ""),
        (r"'can\'t'", "can't"),
        (r'"say \"hello\""', 'say "hello"'),
        (r"'back\\slash'", "back\\slash"),
        (r"'line\nnext'", "line\nnext"),
        (r"'tab\tstop'", "tab\tstop"),
        (r"'\011\u123a\%'", "\t\u123a%"),
    ],
)
def test_dense_quoted_fields_preserve_the_following_target(value: str, expected: str) -> None:
    dataset = stream.iter_arff(io.StringIO(HEADER + f"1,{value},positive\n"), target="label")

    assert list(dataset) == [({"amount": 1.0, "text": expected}, "positive")]


@pytest.mark.parametrize("quote", ["'", '"'])
def test_dense_quoted_nominal_values_are_unquoted(quote: str) -> None:
    header = HEADER.replace("{negative,positive}", "{'class A','class,B'}")
    dataset = stream.iter_arff(
        io.StringIO(header + f"1,plain,{quote}class,B{quote}\n"), target="label"
    )

    assert list(dataset) == [({"amount": 1.0, "text": "plain"}, "class,B")]


def test_dense_fields_can_mix_quote_styles_and_cast_a_quoted_numeric_target() -> None:
    dataset = stream.iter_arff(
        io.StringIO(HEADER + "'1',\"hello, world\",'positive'\n"), target="amount"
    )

    assert list(dataset) == [({"text": "hello, world", "label": "positive"}, 1.0)]


@pytest.mark.parametrize(
    "row", [" 1 , plain , positive ", "\t1\t,\t'plain'\t,\tpositive\t", '1, "plain" , positive']
)
def test_dense_values_ignore_padding(row: str) -> None:
    assert list(stream.iter_arff(io.StringIO(HEADER + row + "\n"), target="label")) == [
        ({"amount": 1.0, "text": "plain"}, "positive")
    ]


def test_dense_blank_and_comment_lines_do_not_yield_observations() -> None:
    rows = "\n \t\n% ignored\n  % ignored\n1,plain,positive\n\n2,other,negative % ignored\n"

    assert list(stream.iter_arff(io.StringIO(HEADER + rows), target="label")) == [
        ({"amount": 1.0, "text": "plain"}, "positive"),
        ({"amount": 2.0, "text": "other"}, "negative"),
    ]


@pytest.mark.parametrize("target", ["amount", "label", ["amount", "label"], None])
def test_dense_padded_missing_values_keep_target_semantics(target: str | list[str] | None) -> None:
    dataset = stream.iter_arff(io.StringIO(HEADER + " ? , ? , ? \n"), target=target)
    expected: typing.Any = {"amount": None, "label": None} if isinstance(target, list) else None

    assert list(dataset) == [({}, expected)]


@pytest.mark.parametrize("weight", ["{5}", " { 0.5 } ", "{1e-3} % ignored"])
def test_dense_instance_weights_do_not_become_features(weight: str) -> None:
    dataset = stream.iter_arff(
        io.StringIO(HEADER + f"1,'hello, world',positive,{weight}\n"), target="label"
    )

    assert list(dataset) == [({"amount": 1.0, "text": "hello, world"}, "positive")]


@pytest.mark.parametrize(
    "row",
    [
        "1,plain",
        "1,plain,positive,extra",
        "1,'unterminated,positive",
        '1,"unterminated,positive',
        "1,'plain'junk,positive",
        "1,plain text,positive",
        "1,plain,positive,{bogus}",
        "1,plain,positive,{5},extra",
        "1,plain,positive,",
        "1,'plain' 'other',positive",
        r"1,'\x00',positive",
    ],
)
def test_malformed_dense_rows_raise_instead_of_misaligning_fields(row: str) -> None:
    with pytest.raises(ValueError):
        list(stream.iter_arff(io.StringIO(HEADER + row + "\n")))
