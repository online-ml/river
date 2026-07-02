"""
The tests performed here confirm the behavioral contracts of preprocessing.GapEncoder: fuzzy
variants of the same string cluster onto the same topic, the vocabulary grows online,
transform_one is read-only, and results are reproducible under a fixed seed.

References
----------
[^1]: Cerda, P. and Varoquaux, G., 2020. Encoding high-cardinality string categorical variables.
    IEEE Transactions on Knowledge and Data Engineering.
    https://inria.hal.science/hal-02171256v4

"""

from __future__ import annotations

import numpy as np

from river import preprocessing

CITY_SET = [
    "london",
    "London",
    "London, UK",
    "Lomdon",
    "paris",
    "Paris",
    "Paris, France",
    "pqris",
]


def _train_city_encoder(n_components=2, **params):
    enc = preprocessing.GapEncoder(n_components=n_components, seed=42, **params)
    for _ in range(10):
        for x in CITY_SET:
            enc.learn_one(x)
    return enc


def test_fuzzy_variants_cluster():
    """Morphological variants of the same city, including unseen typos, activate the same
    topic, and the two cities activate different topics."""
    enc = _train_city_encoder()

    def argmax(x):
        topics = enc.transform_one(x)
        return max(topics, key=topics.get)

    london_topics = {argmax(x) for x in ["london", "London, UK", "Lomdon", "Lndon"]}
    paris_topics = {argmax(x) for x in ["paris", "Paris, France", "pqris", "Pariss"]}

    assert len(london_topics) == 1
    assert len(paris_topics) == 1
    assert london_topics != paris_topics


def test_transform_one_is_pure():
    """transform_one never mutates the model, even on strings with unseen n-grams, and is
    idempotent."""
    enc = _train_city_encoder()

    vocab_size = len(enc.vocab)
    W, A, B = enc.W.copy(), enc.A.copy(), enc.B.copy()

    # Mix of known ("london") and unseen ("zzz") n-grams.
    first = enc.transform_one("londonzzz")
    second = enc.transform_one("londonzzz")

    assert first == second
    assert len(enc.vocab) == vocab_size
    assert np.array_equal(enc.W, W)
    assert np.array_equal(enc.A, A)
    assert np.array_equal(enc.B, B)


def test_seed_reproducibility():
    """Two encoders with the same seed and stream are interchangeable; a different seed leads
    to different topics."""
    stream = CITY_SET * 5

    enc_a = preprocessing.GapEncoder(n_components=2, seed=1)
    enc_b = preprocessing.GapEncoder(n_components=2, seed=1)
    enc_c = preprocessing.GapEncoder(n_components=2, seed=2)
    for x in stream:
        enc_a.learn_one(x)
        enc_b.learn_one(x)
        enc_c.learn_one(x)

    for x in ["london", "paris", "Lndon"]:
        assert enc_a.transform_one(x) == enc_b.transform_one(x)

    # Same stream, hence same vocabulary and shapes, but the topics themselves differ.
    assert enc_a.W.shape == enc_c.W.shape
    assert not np.allclose(enc_a.W, enc_c.W)


def test_unseen_input_transforms_to_zeros():
    """Inputs without any known n-gram, including before any learning, map to all-zero
    activations."""
    enc = preprocessing.GapEncoder(n_components=2, seed=42)
    assert enc.transform_one("anything") == {0: 0.0, 1: 0.0}

    for _ in range(3):
        enc.learn_one("london")

    # No n-gram in common with "london".
    assert enc.transform_one("zzz") == {0: 0.0, 1: 0.0}
    assert enc.transform_one("") == {0: 0.0, 1: 0.0}


def test_vocabulary_growth():
    """The vocabulary and W grow with unseen n-grams, and inputs too short to produce any
    n-gram are no-ops."""
    enc = preprocessing.GapEncoder(n_components=2, ngram_range=(2, 2), seed=42)

    enc.learn_one("ab")
    assert len(enc.vocab) == 1
    assert enc.W.shape == (2, 1)

    enc.learn_one("abc")  # "ab" is known, "bc" is new
    assert len(enc.vocab) == 2
    assert enc.W.shape == (2, 2)

    W, A, B = enc.W.copy(), enc.A.copy(), enc.B.copy()
    enc.learn_one("")
    enc.learn_one("a")
    assert len(enc.vocab) == 2
    assert np.array_equal(enc.W, W)
    assert np.array_equal(enc.A, A)
    assert np.array_equal(enc.B, B)


def test_on_param_matches_plain_strings():
    """With `on` set, the encoder reads the text from a dict field and behaves exactly like
    the plain string encoder."""
    enc_str = preprocessing.GapEncoder(n_components=2, seed=7)
    enc_dict = preprocessing.GapEncoder(on="city", n_components=2, seed=7)

    for _ in range(3):
        for x in CITY_SET:
            enc_str.learn_one(x)
            enc_dict.learn_one({"city": x})

    for x in ["london", "Paris, France", "Lndon"]:
        assert enc_str.transform_one(x) == enc_dict.transform_one({"city": x})


def test_output_keys_and_positivity():
    """Transforming a seen string yields one strictly positive activation per component."""
    enc = _train_city_encoder(n_components=3)
    topics = enc.transform_one("london")

    assert set(topics.keys()) == {0, 1, 2}
    assert all(value > 0 for value in topics.values())


def test_strip_accents_and_lowercase():
    """Default preprocessing folds case and accents before n-gram extraction, so casing and
    accent variants encode identically; with `lowercase=False` case is preserved, leaving an
    all-caps query with no n-gram in common with a lowercase-trained vocabulary."""
    enc = _train_city_encoder()
    london = enc.transform_one("london")

    assert enc.transform_one("LONDON") == london  # case folds onto the trained "london"
    assert enc.transform_one("Lôndon") == london  # accents strip to the same n-grams

    cased = preprocessing.GapEncoder(n_components=2, lowercase=False, seed=42)
    for _ in range(5):
        cased.learn_one("london")

    # Case is kept, so uppercase shares no n-gram with the lowercase vocabulary.
    assert cased.transform_one("LONDON") == {0: 0.0, 1: 0.0}
    assert cased.transform_one("london") != {0: 0.0, 1: 0.0}
