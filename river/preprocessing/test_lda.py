"""
The tests performed here confirm that the outputs of the online preprocessing.LDA are exactly the
same as those of the original with a batch_size of size 1. Coverage is 100%.

References
----------
[^1]: Jordan Boyd-Graber, Ke Zhai, Online Latent Dirichlet Allocation with Infinite Vocabulary.
    http://proceedings.mlr.press/v28/zhai13.pdf
[^2]: river's LDA implementation reproduces exactly the same results as the original one from
    PyInfVov (https://github.com/kzhai/PyInfVoc) with a batch size of 1.

"""
from __future__ import annotations

import numpy as np

from river import preprocessing

DOC_SET = [
    "weather cold",
    "weather hot dry",
    "weather cold rainny",
    "weather hot",
    "weather cold humid",
]

REFERENCE_STATISTICS_TWO_COMPONENTS = [
    {0: np.array([0.0, 0.0, 0.0]), 1: np.array([0.0, 1.0, 1.0])},
    {0: np.array([0.0, 0.6, 0.0, 0.8, 0.6]), 1: np.array([0.0, 0.4, 0.0, 0.2, 0.4])},
    {
        0: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        1: np.array([0.0, 1.0, 1.0, 0.0, 0.0, 1.0]),
    },
    {
        0: np.array([0.0, 0.2, 0.0, 0.6, 0.0, 0.0]),
        1: np.array([0.0, 0.8, 0.0, 0.4, 0.0, 0.0]),
    },
    {
        0: np.array([0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.2]),
        1: np.array([0.0, 1.0, 0.8, 0.0, 0.0, 0.0, 0.8]),
    },
]


REFERENCE_STATISTICS_FIVE_COMPONENTS = [
    {
        0: np.array([0.0, 0.4, 0.2]),
        1: np.array([0.0, 0.2, 0.6]),
        2: np.array([0.0, 0.4, 0.0]),
        3: np.array([0.0, 0.0, 0.0]),
        4: np.array([0.0, 0.0, 0.2]),
    },
    {
        0: np.array([0.0, 0.8, 0.0, 0.4, 0.6]),
        1: np.array([0.0, 0.0, 0.0, 0.2, 0.4]),
        2: np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        3: np.array([0.0, 0.0, 0.0, 0.2, 0.0]),
        4: np.array([0.0, 0.2, 0.0, 0.2, 0.0]),
    },
    {
        0: np.array([0.0, 0.4, 0.2, 0.0, 0.0, 0.0]),
        1: np.array([0.0, 0.2, 0.6, 0.0, 0.0, 0.6]),
        2: np.array([0.0, 0.4, 0.2, 0.0, 0.0, 0.4]),
        3: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        4: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    },
    {
        0: np.array([0.0, 0.2, 0.0, 0.4, 0.0, 0.0]),
        1: np.array([0.0, 0.2, 0.0, 0.2, 0.0, 0.0]),
        2: np.array([0.0, 0.4, 0.0, 0.2, 0.0, 0.0]),
        3: np.array([0.0, 0.2, 0.0, 0.2, 0.0, 0.0]),
        4: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    },
    {
        0: np.array([0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.2]),
        1: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        2: np.array([0.0, 0.8, 0.8, 0.0, 0.0, 0.0, 0.6]),
        3: np.array([0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.2]),
        4: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    },
]

REFERENCE_FIVE_COMPONENTS = [
    np.array([1.5, 0.5, 0.5, 0.5, 1.5]),
    np.array([1.5, 1.5, 0.5, 0.5, 1.5]),
    np.array([0.5, 0.5, 3.5, 0.5, 0.5]),
    np.array([0.5, 0.5, 0.5, 2.5, 0.5]),
    np.array([2.5, 0.5, 0.5, 1.5, 0.5]),
]

REFERENCE_COMPONENTS_WITH_PRUNNING = [
    np.array([0.5, 2.5]),
    np.array([3.5, 0.5]),
    np.array([0.5, 3.5]),
    np.array([2.5, 0.5]),
    np.array([2.5, 1.5]),
]

REFERENCE_LEARN_ONE_PREDICT_ONE = [
    np.array([2.5, 0.5]),
    np.array([2.5, 1.5]),
    np.array([1.5, 2.5]),
    np.array([0.5, 2.5]),
    np.array([0.5, 3.5]),
]


def test_extraction_words_ids():
    """
    Assert that input words are split.
    Assert that indexes are updated and extractable.
    """

    lda = preprocessing.LDA(2, number_of_documents=5, seed=42)

    word_indexes_list = []

    for doc in DOC_SET:
        words = doc.split(" ")

        lda._update_indexes(word_list=words)

        word_indexes_list.append([lda.word_to_index[word] for word in words])

    assert word_indexes_list == [[1, 2], [1, 3, 4], [1, 2, 5], [1, 3], [1, 2, 6]]


def test_statistics_two_components():
    """
    Assert that online lda extracts waited statistics on current document.
    """
    n_components = 2

    lda = preprocessing.LDA(n_components, number_of_documents=60, seed=42)

    statistics_list = []

    for doc in DOC_SET:
        word_list = doc.split(" ")

        lda._update_indexes(word_list=word_list)

        word_indexes = [lda.word_to_index[word] for word in word_list]

        statistics, _ = lda._compute_statistics_components(words_indexes_list=word_indexes)

        statistics_list.append(statistics)

        lda._update_weights(statistics=statistics)

    for index, statistics in enumerate(statistics_list):
        for component in range(n_components):
            assert np.array_equal(
                a1=statistics[component],
                a2=REFERENCE_STATISTICS_TWO_COMPONENTS[index][component],
            )


def test_statistics_five_components():
    """
    Assert that online lda extracts waited statistics on current document.
    """

    n_components = 5

    lda = preprocessing.LDA(
        n_components=n_components,
        number_of_documents=60,
        maximum_size_vocabulary=100,
        alpha_beta=100,
        alpha_theta=0.5,
        seed=42,
    )

    statistics_list = []

    for doc in DOC_SET:
        word_list = doc.split(" ")

        lda._update_indexes(word_list=word_list)

        word_indexes = [lda.word_to_index[word] for word in word_list]

        statistics, _ = lda._compute_statistics_components(words_indexes_list=word_indexes)

        statistics_list.append(statistics)

        lda._update_weights(statistics=statistics)

    for index, statistics in enumerate(statistics_list):
        for component in range(n_components):
            assert np.array_equal(
                a1=statistics[component],
                a2=REFERENCE_STATISTICS_FIVE_COMPONENTS[index][component],
            )


def test_five_components():
    """
    Assert that components computed are identical to the original version for n dimensions.
    """

    n_components = 5

    lda = preprocessing.LDA(
        n_components=n_components,
        number_of_documents=60,
        maximum_size_vocabulary=100,
        alpha_beta=100,
        alpha_theta=0.5,
        seed=42,
    )

    components_list = []

    for document in DOC_SET:
        tokens = {token: 1 for token in document.split(" ")}
        components_list.append(lda.learn_transform_one(tokens))

    for index, component in enumerate(components_list):
        assert np.array_equal(a1=list(component.values()), a2=REFERENCE_FIVE_COMPONENTS[index])


def test_prunning_vocabulary():
    """
    Vocabulary prunning is available to improve accuracy and limit memory usage.
    You can perform vocabulary prunning with parameters vocab_prune_interval (int) and
    maximum_size_vocabulary (int).
    """

    lda = preprocessing.LDA(
        n_components=2,
        number_of_documents=60,
        vocab_prune_interval=2,
        maximum_size_vocabulary=3,
        seed=42,
    )

    components_list = []

    for document in DOC_SET:
        tokens = {token: 1 for token in document.split(" ")}
        components_list.append(lda.learn_transform_one(tokens))

    for index, component in enumerate(components_list):
        assert np.array_equal(
            a1=list(component.values()), a2=REFERENCE_COMPONENTS_WITH_PRUNNING[index]
        )


def test_learn_transform():
    """
    Assert that learn_one and transform_one methods returns waited output.
    """

    lda = preprocessing.LDA(
        n_components=2,
        number_of_documents=60,
        vocab_prune_interval=2,
        maximum_size_vocabulary=3,
        seed=42,
    )
    components_list = []

    for document in DOC_SET:
        tokens = {token: 1 for token in document.split(" ")}
        lda = lda.learn_one(x=tokens)

        components_list.append(lda.transform_one(x=tokens))

    for index, component in enumerate(components_list):
        assert np.array_equal(
            a1=list(component.values()), a2=REFERENCE_LEARN_ONE_PREDICT_ONE[index]
        )
