from __future__ import annotations

import pandas as pd
import pandas.testing as pdt
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

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


def _dense_transform_many(transformer, X):
    return transformer.transform_many(X).sparse.to_dense().sort_index(axis=1)


def _dense_transform_one(transformer, X):
    if transformer.on is None:
        records = X
    else:
        records = X.to_dict(orient="records")

    return (
        pd.DataFrame([transformer.transform_one(x) for x in records], index=X.index)
        .fillna(0.0)
        .sort_index(axis=1)
    )


def _sklearn_tfidf(X):
    if isinstance(X, pd.DataFrame):
        documents = X["text"]
    else:
        documents = X

    sklearn = TfidfVectorizer(token_pattern=r"(?u)\b\w[\w\-]+\b")
    return pd.DataFrame(
        sklearn.fit_transform(documents).toarray(),
        index=X.index,
        columns=sklearn.get_feature_names_out(),
    ).sort_index(axis=1)


def test_bow_transform_many_accepts_dataframe_with_on():
    X = pd.DataFrame(
        {"text": ["Hello world", "Hello River"]},
        index=["river", "rocks"],
    )
    bow = feature_extraction.BagOfWords(on="text")

    pdt.assert_frame_equal(
        _dense_transform_many(bow, X),
        _dense_transform_one(bow, X),
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "X, params",
    [
        pytest.param(
            pd.Series(
                ["foo bar bat baz", "foo bar spam eggs"],
                index=["first", "second"],
            ),
            {},
            id="series",
        ),
        pytest.param(
            pd.DataFrame(
                {"text": ["foo bar bat baz", "foo bar spam eggs"]},
                index=["first", "second"],
            ),
            {"on": "text"},
            id="dataframe-on",
        ),
    ],
)
def test_tfidf_transform_many_matches_transform_one(X, params):
    tfidf = feature_extraction.TFIDF(**params)
    tfidf.learn_many(X)

    pdt.assert_frame_equal(
        _dense_transform_many(tfidf, X),
        _dense_transform_one(tfidf, X),
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "X, params",
    [
        pytest.param(
            pd.Series(
                ["foo foo bar", "foo baz", "spam spam eggs"],
                index=["a", "b", "c"],
            ),
            {},
            id="series",
        ),
        pytest.param(
            pd.DataFrame(
                {"text": ["foo foo bar", "foo baz", "spam spam eggs"]},
                index=["a", "b", "c"],
            ),
            {"on": "text"},
            id="dataframe-on",
        ),
    ],
)
def test_tfidf_transform_many_matches_sklearn_when_normalized(X, params):
    tfidf = feature_extraction.TFIDF(**params)
    tfidf.learn_many(X)

    pdt.assert_frame_equal(
        _dense_transform_many(tfidf, X),
        _sklearn_tfidf(X),
        check_dtype=False,
    )


def test_tfidf_learn_many_matches_learn_one_with_on():
    X = pd.DataFrame(
        {"text": ["foo foo bar", "foo baz", ""]},
        index=["a", "b", "c"],
    )
    batch = feature_extraction.TFIDF(on="text")
    online = feature_extraction.TFIDF(on="text")

    batch.learn_many(X)
    for x in X.to_dict(orient="records"):
        online.learn_one(x)

    assert batch.n == online.n
    assert batch.dfs == online.dfs


def test_tfidf_transform_many_without_normalization():
    X = pd.Series(["foo foo bar", "foo baz"], index=["a", "b"])
    tfidf = feature_extraction.TFIDF(normalize=False)
    tfidf.learn_many(X)

    pdt.assert_frame_equal(
        _dense_transform_many(tfidf, X),
        _dense_transform_one(tfidf, X),
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "transformer, X",
    [
        pytest.param(
            feature_extraction.BagOfWords(),
            pd.Series([""], index=["empty"]),
            id="bow-series",
        ),
        pytest.param(
            feature_extraction.BagOfWords(on="text"),
            pd.DataFrame({"text": [""]}, index=["empty"]),
            id="bow-dataframe-on",
        ),
        pytest.param(
            feature_extraction.TFIDF(),
            pd.Series([""], index=["empty"]),
            id="tfidf-series",
        ),
        pytest.param(
            feature_extraction.TFIDF(on="text"),
            pd.DataFrame({"text": [""]}, index=["empty"]),
            id="tfidf-dataframe-on",
        ),
    ],
)
def test_vectorizers_transform_many_accept_empty_documents(transformer, X):
    Xt = transformer.transform_many(X)

    assert Xt.index.tolist() == ["empty"]
    assert Xt.shape == (1, 0)
