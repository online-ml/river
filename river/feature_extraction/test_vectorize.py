from __future__ import annotations

import math
import typing

import narwhals.stable.v2 as nw
import pandas as pd
import pandas.testing as pdt
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from river import feature_extraction
from river.conftest import FRAME_BACKENDS

if typing.TYPE_CHECKING:
    from river.conftest import FrameBackend


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


# Dataframe-agnostic mini-batching (narwhals): the tests above pin down pandas behaviour; the
# ones below re-run the core equivalences across every supported backend (pandas, polars,
# pyarrow, ...) via the `frame_backend` fixture from `river/conftest.py`.

DOCS = ["foo foo bar", "foo baz", "spam spam eggs", ""]


def _make_input(backend: FrameBackend, docs: list[str], *, on: str | None):
    """Build the native input: a series of documents when `on is None`, else a dataframe."""
    return backend.series(docs) if on is None else backend.frame({on: docs})


def _native_columns(native) -> dict[str, list[float]]:
    """Normalise any backend's `transform_many` output to `{str(term): [float, ...]}`."""
    frame = nw.from_native(native, eager_only=True)
    return {str(col): [float(v) for v in frame[col].to_list()] for col in frame.columns}


def _one_columns(transformer, docs: list[str], *, on: str | None) -> dict[str, list[float]]:
    """The same mapping, built from a row-by-row `transform_one` loop (the reference)."""
    rows = [transformer.transform_one(doc if on is None else {on: doc}) for doc in docs]
    terms = {term for row in rows for term in row}
    return {str(term): [float(row.get(term, 0.0)) for row in rows] for term in terms}


def _assert_columns_close(actual: dict, expected: dict) -> None:
    assert actual.keys() == expected.keys()
    for term, values in expected.items():
        assert all(
            math.isclose(a, e, rel_tol=1e-9, abs_tol=1e-12) for a, e in zip(actual[term], values)
        )


@pytest.mark.parametrize("on", [None, "text"])
@pytest.mark.parametrize("normalize", [True, False])
def test_tfidf_transform_many_matches_transform_one_across_backends(
    frame_backend: FrameBackend, on: str | None, normalize: bool
) -> None:
    """`transform_many` equals a row-by-row `transform_one` loop on every backend."""
    X = _make_input(frame_backend, DOCS, on=on)

    tfidf = feature_extraction.TFIDF(normalize=normalize, on=on)
    tfidf.learn_many(X)

    _assert_columns_close(
        _native_columns(tfidf.transform_many(X)),
        _one_columns(tfidf, DOCS, on=on),
    )


@pytest.mark.parametrize("on", [None, "text"])
def test_bow_transform_many_matches_transform_one_across_backends(
    frame_backend: FrameBackend, on: str | None
) -> None:
    """`BagOfWords.transform_many` equals a `transform_one` loop on every backend (issue #1576)."""
    X = _make_input(frame_backend, DOCS, on=on)
    bow = feature_extraction.BagOfWords(on=on)

    _assert_columns_close(
        _native_columns(bow.transform_many(X)),
        _one_columns(bow, DOCS, on=on),
    )


@pytest.mark.parametrize("on", [None, "text"])
def test_tfidf_transform_many_matches_sklearn_across_backends(
    frame_backend: FrameBackend, on: str | None
) -> None:
    """Normalised `transform_many` matches scikit-learn's `TfidfVectorizer` on every backend."""
    docs = ["foo foo bar", "foo baz", "spam spam eggs"]
    X = _make_input(frame_backend, docs, on=on)

    tfidf = feature_extraction.TFIDF(on=on)
    tfidf.learn_many(X)
    actual = _native_columns(tfidf.transform_many(X))

    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w[\w\-]+\b")
    matrix = vectorizer.fit_transform(docs).toarray()
    expected = {
        term: [float(matrix[i, j]) for i in range(len(docs))]
        for j, term in enumerate(vectorizer.get_feature_names_out())
    }

    _assert_columns_close(actual, expected)


@pytest.mark.parametrize("on", [None, "text"])
def test_tfidf_learn_many_matches_learn_one_across_backends(
    frame_backend: FrameBackend, on: str | None
) -> None:
    """`learn_many` accumulates the same document frequencies as a `learn_one` loop."""
    X = _make_input(frame_backend, DOCS, on=on)

    batch = feature_extraction.TFIDF(on=on)
    batch.learn_many(X)

    online = feature_extraction.TFIDF(on=on)
    for doc in DOCS:
        online.learn_one(doc if on is None else {on: doc})

    assert batch.n == online.n
    assert batch.dfs == online.dfs


@pytest.mark.parametrize(
    "transformer",
    [feature_extraction.BagOfWords(on="text"), feature_extraction.TFIDF(on="text")],
    ids=["bow", "tfidf"],
)
def test_transform_many_returns_input_backend(frame_backend: FrameBackend, transformer) -> None:
    """The output frame's native type matches the input dataframe's backend."""
    X = frame_backend.frame({"text": DOCS})
    if isinstance(transformer, feature_extraction.TFIDF):
        transformer.learn_many(X)

    assert type(transformer.transform_many(X)) is type(X)


def test_tfidf_learn_many_is_backend_agnostic(frame_backend: FrameBackend) -> None:
    """`learn_many` accumulates identical state regardless of the input backend."""
    reference = feature_extraction.TFIDF(on="text")
    reference.learn_many(FRAME_BACKENDS["pandas"]().frame({"text": DOCS}))

    model = feature_extraction.TFIDF(on="text")
    model.learn_many(frame_backend.frame({"text": DOCS}))

    assert model.n == reference.n
    assert model.dfs == reference.dfs
