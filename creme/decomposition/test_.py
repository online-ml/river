"""
The tests performed here confirm that the outputs of the online LDA are exactly the same as those of
the original with a batch_size of size 1. Coverage is 100%.

References:
    1. Jordan Boyd-Graber, Ke Zhai, Online Latent Dirichlet Allocation with Infinite Vocabulary.
    http://proceedings.mlr.press/v28/zhai13.pdf

    2. Creme's Online LDA reproduces exactly the same results of the original one with a size of
    batch 1:
    https://github.com/kzhai/PyInfVoc.
"""
import numpy as np

from creme.decomposition import LDA


DOC_SET = [
    'weather cold',
    'weather hot dry',
    'weather cold rainny',
    'weather hot',
    'weather cold humid',
]

REFERENCE_STATISTICS_TWO_COMPONENTS = [
    {
        0: np.array([0., 0., 0.]),
        1: np.array([0., 1., 1.]),
    },
    {
        0: np.array([0., 0.6, 0., 0.8, 0.6]),
        1: np.array([0., 0.4, 0., 0.2, 0.4])},
    {
        0: np.array([0., 0., 0., 0., 0., 0.]),
        1: np.array([0., 1., 1., 0., 0., 1.])},
    {
        0: np.array([0., 0.2, 0., 0.6, 0., 0.]),
        1: np.array([0., 0.8, 0., 0.4, 0., 0.])},
    {
        0: np.array([0., 0., 0.2, 0., 0., 0., 0.2]),
        1: np.array([0., 1., 0.8, 0., 0., 0., 0.8]),
    },
]


REFERENCE_STATISTICS_FIVE_COMPONENTS = [
    {
        0: np.array([0., 0.4, 0.2]),
        1: np.array([0., 0.2, 0.6]),
        2: np.array([0., 0.4, 0.]),
        3: np.array([0., 0., 0.]),
        4: np.array([0., 0., 0.2])
    },
    {
        0: np.array([0., 0.8, 0., 0.4, 0.6]),
        1: np.array([0., 0., 0., 0.2, 0.4]),
        2: np.array([0., 0., 0., 0., 0.]),
        3: np.array([0., 0., 0., 0.2, 0.]),
        4: np.array([0., 0.2, 0., 0.2, 0.])
    },
    {
        0: np.array([0., 0.4, 0.2, 0., 0., 0.]),
        1: np.array([0., 0.2, 0.6, 0., 0., 0.6]),
        2: np.array([0., 0.4, 0.2, 0., 0., 0.4]),
        3: np.array([0., 0., 0., 0., 0., 0.]),
        4: np.array([0., 0., 0., 0., 0., 0.])
    },
    {
        0: np.array([0., 0.2, 0., 0.4, 0., 0.]),
        1: np.array([0., 0.2, 0., 0.2, 0., 0.]),
        2: np.array([0., 0.4, 0., 0.2, 0., 0.]),
        3: np.array([0., 0.2, 0., 0.2, 0., 0.]),
        4: np.array([0., 0., 0., 0., 0., 0.])
    },
    {
        0: np.array([0., 0., 0.2, 0., 0., 0., 0.2]),
        1: np.array([0., 0., 0., 0., 0., 0., 0.]),
        2: np.array([0., 0.8, 0.8, 0., 0., 0., 0.6]),
        3: np.array([0., 0.2, 0., 0., 0., 0., 0.2]),
        4: np.array([0., 0., 0., 0., 0., 0., 0.])
    }
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

REFERENCE_FIT_ONE_PREDICT_ONE = [
    np.array([2.5, 0.5]),
    np.array([2.5, 1.5]),
    np.array([1.5, 2.5]),
    np.array([0.5, 2.5]),
    np.array([0.5, 3.5]),
]


def test_extraction_words_ids():
    '''
    Assert that inputs words are splitted.
    Assert that indexes are updated and extractable.
    '''
    np.random.seed(42)

    lda = LDA(2, number_of_documents=5)

    word_indexes_list = []

    for doc in DOC_SET:

        words = lda.tokenizer(lda.preprocess(lda._get_text(doc)))

        lda._update_indexes(word_list=words)

        word_indexes_list.append(
            [lda.word_to_index[word] for word in words]
        )

    assert word_indexes_list == [
        [1, 2],
        [1, 3, 4],
        [1, 2, 5],
        [1, 3],
        [1, 2, 6],
    ]


def test_statistics_two_components():
    '''
    Assert that online lda extracts waited statistics on current document.
    '''
    n_components = 2

    np.random.seed(42)

    lda = LDA(n_components, number_of_documents=60)

    statistics_list = []

    for doc in DOC_SET:

        word_list = lda.tokenizer(lda.preprocess(lda._get_text(doc)))

        lda._update_indexes(word_list=word_list)

        word_indexes = [lda.word_to_index[word] for word in word_list]

        statistics, _ = lda._compute_statistics_components(
            words_indexes_list=word_indexes,
        )

        statistics_list.append(statistics)

        lda._update_weights(
            statistics=statistics
        )

    for index, statistics in enumerate(statistics_list):
        for component in range(n_components):
            assert np.array_equal(
                a1=statistics[component],
                a2=REFERENCE_STATISTICS_TWO_COMPONENTS[index][component],
            )


def test_statistics_five_components():
    '''
    Assert that online lda extracts waited statistics on current document.
    '''
    np.random.seed(42)

    n_components = 5

    lda = LDA(
        n_components=n_components,
        number_of_documents=60,
        maximum_size_vocabulary=100,
        alpha_beta=100,
        alpha_theta=0.5,
    )

    statistics_list = []

    for doc in DOC_SET:

        word_list = lda.tokenizer(lda.preprocess(lda._get_text(doc)))

        lda._update_indexes(word_list=word_list)

        word_indexes = [lda.word_to_index[word] for word in word_list]

        statistics, _ = lda._compute_statistics_components(
            words_indexes_list=word_indexes,
        )

        statistics_list.append(statistics)

        lda._update_weights(
            statistics=statistics
        )

    for index, statistics in enumerate(statistics_list):
        for component in range(n_components):
            assert np.array_equal(
                a1=statistics[component],
                a2=REFERENCE_STATISTICS_FIVE_COMPONENTS[index][component],
            )


def test_five_components():
    '''
    Assert that components computed are identical to the original version for n dimensions.
    '''
    np.random.seed(42)

    n_components = 5

    online_lda = LDA(
        n_components=n_components,
        number_of_documents=60,
        maximum_size_vocabulary=100,
        alpha_beta=100,
        alpha_theta=0.5,
    )

    components_list = []

    for document in DOC_SET:
        components_list.append(online_lda.fit_transform_one(document))

    for index, component in enumerate(components_list):
        assert np.array_equal(
            a1=list(component.values()),
            a2=REFERENCE_FIVE_COMPONENTS[index],
        )


def test_prunning_vocabulary():
    '''
    Vocabulary prunning is available to improve accuracy and limit memory usage.
    You can perform vocabulary prunning with parameters vocab_prune_interval (int) and
    maximum_size_vocabulary (int).
    '''
    np.random.seed(42)

    online_lda = LDA(
        n_components=2,
        number_of_documents=60,
        vocab_prune_interval=2,
        maximum_size_vocabulary=3
    )

    components_list = []

    for document in DOC_SET:
        components_list.append(
            online_lda.fit_transform_one(x=document)
        )

    for index, component in enumerate(components_list):
        assert np.array_equal(
            a1=list(component.values()),
            a2=REFERENCE_COMPONENTS_WITH_PRUNNING[index]
        )


def test_fit_transform():
    '''
    Assert that fit_one and transform_one methods returns waited ouput.
    '''
    np.random.seed(42)

    online_lda = LDA(
        n_components=2,
        number_of_documents=60,
        vocab_prune_interval=2,
        maximum_size_vocabulary=3,
    )
    components_list = []

    for document in DOC_SET:
        online_lda = online_lda.fit_one(x=document)

        components_list.append(
            online_lda.transform_one(x=document)
        )

    for index, component in enumerate(components_list):
        assert np.array_equal(
            a1=list(component.values()),
            a2=REFERENCE_FIT_ONE_PREDICT_ONE[index]
        )
