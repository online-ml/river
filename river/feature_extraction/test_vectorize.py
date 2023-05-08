from __future__ import annotations

import pytest

from river import feature_extraction


@pytest.mark.parametrize(
    "params, text, expected_ngrams",
    [
        pytest.param(
            *case,
            id=f"#{i}",
        )
        for i, case in enumerate(
            [
                ({}, "one two three", ["one", "two", "three"]),
                (
                    {},
                    """one   two\tthree four\t\tfive
            six

            seven""",
                    ["one", "two", "three", "four", "five", "six", "seven"],
                ),
                (
                    {"ngram_range": (1, 2)},
                    "one two three",
                    ["one", "two", "three", ("one", "two"), ("two", "three")],
                ),
                ({"ngram_range": (2, 2)}, "one two three", [("one", "two"), ("two", "three")]),
                (
                    {"ngram_range": (2, 3)},
                    "one two three",
                    [("one", "two"), ("two", "three"), ("one", "two", "three")],
                ),
                ({"stop_words": {"two", "three"}}, "one two three four", ["one", "four"]),
                (
                    {"stop_words": {"two", "three"}, "ngram_range": (1, 2)},
                    "one two three four",
                    ["one", "four", ("one", "four")],
                ),
            ]
        )
    ],
)
def test_ngrams(params, text, expected_ngrams):
    bow = feature_extraction.BagOfWords(**params)
    ngrams = list(bow.process_text(text))
    assert expected_ngrams == ngrams
