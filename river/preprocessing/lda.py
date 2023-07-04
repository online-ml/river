from __future__ import annotations

import functools
import typing
from collections import defaultdict

import numpy as np
from scipy import ndimage, special

from river import base

__all__ = ["LDA"]


class LDA(base.Transformer):
    """Online Latent Dirichlet Allocation with Infinite Vocabulary.

    Latent Dirichlet allocation (LDA) is a probabilistic approach for exploring topics in document
    collections. The key advantage of this variant is that it assumes an infinite vocabulary,
    meaning that the set of tokens does not have to known in advance, as opposed to the
    [implementation from sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html)
    The results produced by this implementation are identical to those from [the original
    implementation](https://github.com/kzhai/PyInfVoc) proposed by the method's authors.

    This class takes as input token counts. Therefore, it requires you to tokenize beforehand. You
    can do so by using a `feature_extraction.BagOfWords` instance, as shown in the example below.

    Parameters
    ----------
    n_components
        Number of topics of the latent Drichlet allocation.
    number_of_documents
        Estimated number of documents.
    alpha_theta
        Hyper-parameter of the Dirichlet distribution of topics.
    alpha_beta
        Hyper-parameter of the Dirichlet process of distribution over words.
    tau
        Learning inertia to prevent premature convergence.
    kappa
        The learning rate kappa controls how quickly new parameters estimates replace the old ones.
        kappa ∈ (0.5, 1] is required for convergence.
    vocab_prune_interval
        Interval at which to refresh the words topics distribution.
    number_of_samples
        Number of iteration to computes documents topics distribution.
    ranking_smooth_factor
    burn_in_sweeps
        Number of iteration necessaries while analyzing a document before updating document topics
        distribution.
    maximum_size_vocabulary
        Maximum size of the stored vocabulary.
    seed
        Random number seed used for reproducibility.

    Attributes
    ----------
    counter : int
        The current number of observed documents.
    truncation_size_prime : int
        Number of distincts words stored in the vocabulary. Updated before processing a document.
    truncation_size : int
        Number of distincts words stored in the vocabulary. Updated after processing a document.
    word_to_index : dict
        Words as keys and indexes as values.
    index_to_word : dict
        Indexes as keys and words as values.
    nu_1 : dict
        Weights of the words. Component of the variational inference.
    nu_2 : dict
        Weights of the words. Component of the variational inference.

    Examples
    --------

    >>> from river import compose
    >>> from river import feature_extraction
    >>> from river import preprocessing

    >>> X = [
    ...    'weather cold',
    ...    'weather hot dry',
    ...    'weather cold rainy',
    ...    'weather hot',
    ...    'weather cold humid',
    ... ]

    >>> lda = compose.Pipeline(
    ...     feature_extraction.BagOfWords(),
    ...     preprocessing.LDA(
    ...         n_components=2,
    ...         number_of_documents=60,
    ...         seed=42
    ...     )
    ... )

    >>> for x in X:
    ...     lda = lda.learn_one(x)
    ...     topics = lda.transform_one(x)
    ...     print(topics)
    {0: 0.5, 1: 2.5}
    {0: 2.499..., 1: 1.5}
    {0: 0.5, 1: 3.5}
    {0: 0.5, 1: 2.5}
    {0: 1.5, 1: 2.5}

    References
    ----------
    [^1]: [Zhai, K. and Boyd-Graber, J., 2013, February. Online latent Dirichlet allocation with infinite vocabulary. In International Conference on Machine Learning (pp. 561-569).](http://proceedings.mlr.press/v28/zhai13.pdf)
    [^2]: [PyInfVoc on GitHub](https://github.com/kzhai/PyInfVoc)

    """

    def __init__(
        self,
        n_components=10,
        number_of_documents=1e6,
        alpha_theta=0.5,
        alpha_beta=100.0,
        tau=64.0,
        kappa=0.75,
        vocab_prune_interval=10,
        number_of_samples=10,
        ranking_smooth_factor=1e-12,
        burn_in_sweeps=5,
        maximum_size_vocabulary=4000,
        seed: int | None = None,
    ):
        self.n_components = n_components
        self.number_of_documents = number_of_documents
        self.alpha_theta = alpha_theta
        self.alpha_beta = alpha_beta
        self.tau = tau
        self.kappa = kappa
        self.vocab_prune_interval = vocab_prune_interval
        self.number_of_samples = number_of_samples
        self.ranking_smooth_factor = ranking_smooth_factor
        self.burn_in_sweeps = burn_in_sweeps
        self.maximum_size_vocabulary = maximum_size_vocabulary
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.counter = 0
        self.truncation_size_prime = 1
        self.truncation_size = 1

        self.word_to_index: dict[str, int] = {}
        self.index_to_word: dict[int, str] = {}

        self.nu_1: defaultdict = defaultdict(functools.partial(np.ones, 1))
        self.nu_2: defaultdict = defaultdict(functools.partial(np.array, [self.alpha_beta]))

        for topic in range(self.n_components):
            self.nu_1[topic] = np.ones(1)
            self.nu_2[topic] = np.array([self.alpha_beta])

    def learn_transform_one(self, x: dict) -> dict:
        """Equivalent to `lda.learn_one(x).transform_one(x)`s, but faster.

        Parameters
        ----------
        x: A document.

        Returns
        -------
        Component attributions for the input document.

        """

        # Updates number of documents:
        self.counter += 1

        # Extracts words of the document as a list of words:
        word_list: typing.Iterable[str] = x.keys()

        # Update words indexes:
        self._update_indexes(word_list=word_list)

        # Replace the words by their index:
        words_indexes_list = [self.word_to_index[word] for word in word_list]

        # Sample empirical topic assignment:
        (
            statistics,
            batch_document_topic_distribution,
        ) = self._compute_statistics_components(words_indexes_list)

        # Online variational inference
        self._update_weights(statistics=statistics)

        if self.counter % self.vocab_prune_interval == 0:
            self._prune_vocabulary()

        return dict(enumerate(batch_document_topic_distribution))

    def learn_one(self, x):
        self.learn_transform_one(x)
        return self

    def transform_one(self, x):
        # Extracts words of the document as a list of words:
        word_list = x.keys()

        # Update words indexes:
        self._update_indexes(word_list=word_list)

        # Replace the words by their index:
        words_indexes_list = [self.word_to_index[word] for word in word_list]

        # Sample empirical topic assignment:
        _, components = self._compute_statistics_components(words_indexes_list)

        return dict(enumerate(components))

    def _update_indexes(self, word_list: typing.Iterable[str]):
        """
        Adds the words of the document to the index if they are not part of the current vocabulary.
        Updates of the number of distinct words seen.

        Parameters
        ----------
        word_list
            Content of the document as a list of words.

        """
        for word in word_list:
            if word not in self.word_to_index:
                new_index = len(self.word_to_index) + 1
                self.word_to_index[word] = new_index
                self.index_to_word[new_index] = word
                self.truncation_size_prime += 1

    @classmethod
    def _compute_weights(cls, n_components: int, nu_1: dict, nu_2: dict) -> tuple[dict, dict]:
        """Calculates the vocabulary weighting according to the word distribution present in the
        vocabulary.

        The Psi function is the logarithmic derivative of the gamma function.

        Parameters
        ----------
        n_components
            Number of topics.
        nu_1
            Weights of the words of the vocabulary.
        nu_2
            Weights of the words of the vocabulary.

        Returns
        -------
        Weights of the words of the current vocabulary.

        """
        exp_weights = {}
        exp_oov_weights = {}

        for topic in range(n_components):
            psi_nu_1 = special.psi(nu_1[topic])
            psi_nu_2 = special.psi(nu_2[topic])

            psi_nu_1_nu_2 = special.psi(nu_1[topic] + nu_2[topic])

            psi_nu_1_nu_2_minus_psi_nu_2 = np.cumsum([psi_nu_2 - psi_nu_1_nu_2], axis=1)

            exp_oov_weights[topic] = np.exp(psi_nu_1_nu_2_minus_psi_nu_2[0][-1])

            psi_nu_1_nu_2_minus_psi_nu_2 = ndimage.shift(
                input=psi_nu_1_nu_2_minus_psi_nu_2[0], shift=1, cval=0
            )

            exp_weights[topic] = np.exp(psi_nu_1 - psi_nu_1_nu_2 + psi_nu_1_nu_2_minus_psi_nu_2)

        return exp_weights, exp_oov_weights

    def _update_weights(self, statistics):
        """Learn documents and word representations. Calculate the variational approximation.

        Parameters
        ----------
        statistics
            Weights associated to the words.

        """
        reverse_cumulated_phi = {}

        for k in range(self.n_components):
            reverse_cumulated_phi[k] = ndimage.shift(input=statistics[k], shift=-1, cval=0)

            reverse_cumulated_phi[k] = np.flip(reverse_cumulated_phi[k])
            reverse_cumulated_phi[k] = np.cumsum(reverse_cumulated_phi[k])
            reverse_cumulated_phi[k] = np.flip(reverse_cumulated_phi[k])

        # Epsilon will be between 0 and 1.
        # Epsilon value says how much to weight the information we got from this document.
        self.epsilon = (self.tau + self.counter) ** -self.kappa

        for k in range(self.n_components):
            if self.truncation_size < self.truncation_size_prime:
                difference_truncation = self.truncation_size_prime - self.truncation_size

                self.nu_1[k] = np.append(self.nu_1[k], np.ones(difference_truncation))
                self.nu_2[k] = np.append(self.nu_2[k], np.ones(difference_truncation))

            # Variational Approximation
            self.nu_1[k] += self.epsilon * (
                self.number_of_documents * np.array(statistics[k]) + 1 - self.nu_1[k]
            )

            self.nu_2[k] += self.epsilon * (
                self.alpha_beta
                + self.number_of_documents * np.array(reverse_cumulated_phi[k])
                - self.nu_2[k]
            )

        self.truncation_size = self.truncation_size_prime

    def _compute_statistics_components(self, words_indexes_list: list) -> tuple[dict, dict]:
        """Extract latent variables from the document and words.

        Parameters
        ----------
        words_indexes_list
            Ids of the words of the input document.

        Returns
        -------
        Computed statistics over the words. Document reprensetation across topics.

        """
        statistics: defaultdict = defaultdict(lambda: np.zeros(self.truncation_size_prime))

        exp_weights, exp_oov_weights = self._compute_weights(
            n_components=self.n_components, nu_1=self.nu_1, nu_2=self.nu_2
        )

        size_vocab = len(words_indexes_list)

        phi = self.rng.random((self.n_components, size_vocab))

        phi /= np.sum(phi, axis=0)

        phi_sum = np.sum(phi, axis=1)

        for sample_index in range(self.number_of_samples):
            for word_index in range(size_vocab):
                phi_sum -= phi[:, word_index]
                phi_sum = phi_sum.clip(min=0)
                temp_phi = phi_sum + self.alpha_theta

                for k in range(self.n_components):
                    if words_indexes_list[word_index] >= self.truncation_size:
                        temp_phi[k] *= exp_oov_weights[k]
                    else:
                        temp_phi[k] *= exp_weights[k][words_indexes_list[word_index]]

                # Normalize document topic distribution before applying multinomial distribution:
                temp_phi /= temp_phi.sum()

                # Sample a topic based a given probability distribution:
                temp_phi = self.rng.multinomial(1, temp_phi)

                phi[:, word_index] = temp_phi

                phi_sum += temp_phi

                if sample_index >= self.burn_in_sweeps:
                    for k in range(self.n_components):
                        index = words_indexes_list[word_index]

                        statistics[k][index] += temp_phi[k]

        document_topic_distribution = self.alpha_theta + phi_sum

        for k in range(self.n_components):
            statistics[k] /= self.number_of_samples - self.burn_in_sweeps

        return statistics, document_topic_distribution

    def _prune_vocabulary(self):
        """Reduce the size of the index exceeds the maximum size."""
        if self.nu_1[0].shape[0] > self.maximum_size_vocabulary:
            for topic in range(self.n_components):
                # Updates words latent variables
                self.nu_1[topic] = self.nu_1[topic][: self.maximum_size_vocabulary]
                self.nu_2[topic] = self.nu_2[topic][: self.maximum_size_vocabulary]

            new_word_to_index = {}
            new_index_to_word = {}

            for index in range(1, self.maximum_size_vocabulary):
                # Updates words indexes
                word = self.index_to_word[index]
                new_word_to_index[word] = index
                new_index_to_word[index] = word

            self.word_to_index = new_word_to_index
            self.index_to_word = new_index_to_word

            self.truncation_size = self.nu_1[0].shape[0]
            self.truncation_size_prime = self.truncation_size
