from __future__ import annotations

import pickle
import sys

import pytest

from river import misc

zstd_only = pytest.mark.skipif(
    sys.version_info < (3, 14),
    reason="ZstdClassifier requires Python 3.14 (compression.zstd)",
)


def test_requires_python_314():
    if sys.version_info < (3, 14):
        with pytest.raises(RuntimeError, match="Python 3.14"):
            misc.ZstdClassifier()
    else:
        misc.ZstdClassifier()


@zstd_only
def test_learn_and_predict():
    model = misc.ZstdClassifier(window=100_000, level=3, rebuild_every=1)
    animal_corpus = [
        "the cat sat on the mat",
        "a dog barked at the moon",
        "the bird flew over the tree",
        "the kitten purred softly",
        "the puppy played in the garden",
        "hamsters run on wheels",
    ] * 20
    finance_corpus = [
        "stocks rallied after the report",
        "the central bank raised rates",
        "bond yields fell sharply today",
        "treasury yields slipped after the announcement",
        "equity markets surged on positive earnings",
        "dividends were paid to shareholders",
    ] * 20
    for text in animal_corpus:
        model.learn_one(text, "animal")
    for text in finance_corpus:
        model.learn_one(text, "finance")

    assert model.predict_one("the dog chased the cat in the garden") == "animal"
    assert model.predict_one("treasury bond yields slipped on inflation data") == "finance"

    probas = model.predict_proba_one("the dog chased the cat in the garden")
    assert set(probas) == {"animal", "finance"}
    assert abs(sum(probas.values()) - 1.0) < 1e-9
    assert probas["animal"] > probas["finance"]


@zstd_only
def test_empty_predict_returns_empty_dict():
    model = misc.ZstdClassifier()
    assert model.predict_proba_one("anything") == {}
    assert model.predict_one("anything") is None


@zstd_only
def test_sliding_window_eviction():
    model = misc.ZstdClassifier(window=20, rebuild_every=1)
    model.learn_one("a" * 30, "x")
    assert len(model.buffers["x"]) == 20
    model.learn_one("b" * 5, "x")
    assert len(model.buffers["x"]) == 20
    assert model.buffers["x"].endswith(b"bbbbb")


@zstd_only
def test_on_extracts_field():
    model = misc.ZstdClassifier(window=4096, rebuild_every=1, on="text")
    model.learn_one({"text": "cats and dogs"}, "animal")
    assert b"cats and dogs" in bytes(model.buffers["animal"])


@zstd_only
def test_pickling_roundtrip():
    model = misc.ZstdClassifier(rebuild_every=1)
    model.learn_one("cats and dogs", "animal")
    model.learn_one("stocks rose today", "finance")
    # Trigger compressor build
    model.predict_one("dogs")
    restored = pickle.loads(pickle.dumps(model))
    assert dict(restored.buffers) == {
        "animal": bytearray(b"cats and dogs"),
        "finance": bytearray(b"stocks rose today"),
    }
    assert restored.predict_one("dogs") in {"animal", "finance"}


@zstd_only
def test_clone_resets_state():
    model = misc.ZstdClassifier()
    model.learn_one("hello", "a")
    clone = model.clone()
    assert clone.buffers == {}
