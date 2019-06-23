# Python
from collections import defaultdict

# Third-parties:
import numpy as np

from scipy import special
from scipy import ndimage

__all__ = ['OnlineLda']


class OnlineLda:
    """Online Latent Dirichlet Allocation with Infinite Vocabulary.

    Latent Dirichlet allocation (LDA) is a probabilistic approach for exploring topics in document
    collections.

    Args:
        n_components (int): Number of topics of the latent Drichlet allocation.
        number_of_documents (int): Estimated number of documents.
        alpha_theta (float): Hyper-parameter for Dirichlet distribution of topics,
            1.0/n_components in the paper.
        alpha_beta (float): Hyper-parameter for Dirichlet process of distribution over words 1e3
            in the paper.
        tau (float): The learning inertia tau prevents premature convergence.
        kappa (float): The learning rate kappa controls how quickly new parameters estimates replace
            the old ones. kappa ∈ (0.5, 1] is required for convergence.
        vocab_prune_interval (int): Interval to refresh the words topics distribution.
        number_of_samples (int): Number of iteration to computes documents topics distribution.
        burn_in_sweeps (int): Number of iteration necessaries while analyzing a document
             before updating document topics distribution.
        maximum_size_vocabulary (int): Maximum size of the vocabulary stored.

    Attributes:
        n_components (int): Number of topics of the latent Drichlet allocation.
        number_of_documents (int): Estimated number of documents.
        alpha_theta (float): Hyper-parameter for Dirichlet distribution of topics,
            1.0/n_components in the paper.
        alpha_beta (float): Hyper-parameter for Dirichlet process of distribution over words 1e3
            in the paper.
        tau (float): The learning inertia tau prevents premature convergence.
        kappa (float): The learning rate kappa controls how quickly new parameters estimates replace
            the old ones. kappa ∈ (0.5, 1] is required for convergence.
        vocab_prune_interval (int): Interval to refresh the words topics distribution.
        number_of_samples (int): Number of iteration to computes documents topics distribution.
        burn_in_sweeps (int): Number of iteration necessaries while analyzing a document
             before updating document topics distribution.
        maximum_size_vocabulary (int): Maximum size of the vocabulary stored.
        counter (int): Number of documents that updated the latent Dirichlet allocation.
        truncation_size_prime (int): Number of distincts words stored in the vocabulary. Updated
            before processing a document.
        truncation_size (int) : Number of distincts words stored in the vocabulary. Updated after
            processing a document.
        word_to_index (dict): Words as keys and indexes as values.
        index_to_word (dict): Indexes as keys and words as values.
        nu_1 (dict): Weights of the words. Component of the variational inference.
        nu_2 (dict): Weights of the words. Component of the variational inference.


        Example:
                ::
                    >>> from creme import decomposition
                    >>> import numpy as np

                    >>> np.random.seed(42)

                    >>> X = [
                    ...    'weather cold',
                    ...    'weather hot dry',
                    ...    'weather cold rainny',
                    ...    'weather hot',
                    ...    'weather cold humid',
                    ... ]

                    >>> online_lda = decomposition.OnlineLda(n_components=2, number_of_documents=60)
                    >>> for x in X:
                    ...     print(online_lda.fit_transform_one(x))
                    [0.5 2.5]
                    [3.5 0.5]
                    [0.5 3.5]
                    [1.5 1.5]
                    [2.5 1.5]

    References:

        1. Jordan Boyd-Graber, Ke Zhai, Online Latent Dirichlet Allocation with Infinite Vocabulary.
        http://proceedings.mlr.press/v28/zhai13.pdf

        2. Creme's Online LDA reproduces exactly the same results of the original one with a size of
        batch 1:
        https://github.com/kzhai/PyInfVoc.
    """

    def __init__(
        self,
        n_components,
        number_of_documents,
        alpha_theta=0.5,
        alpha_beta=100.,
        tau=64.,
        kappa=0.75,
        vocab_prune_interval=10,
        number_of_samples=10,
        ranking_smooth_factor=1e-12,
        burn_in_sweeps=5,
        maximum_size_vocabulary=4000,
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

        self.counter = 0
        self.truncation_size_prime = 1
        self.truncation_size = 1

        self.word_to_index = {}
        self.index_to_word = {}

        self.nu_1 = defaultdict(lambda: np.ones(1))
        self.nu_2 = defaultdict(lambda: np.array([self.alpha_beta]))

        for topic in range(self.n_components):
            self.nu_1[topic] = np.ones(1)
            self.nu_2[topic] = np.array([self.alpha_beta])

    def fit_transform_one(self, document):
        """
        Updates of the Dirichlet latent allocation. Splits the words in the document into a list.
        Updates word indexes. Break down the document into components. Performs the variational
        online inference. fit_transform_one is the most effective way to run OnlineLda and to reduce
        calculation costs.

        Args:
            document (str): Current document to be encoded and reduced in size.

        Returns
            list: Components of the input document.
        """
        # Updates number of documents:
        self.counter += 1

        # Extracts words of the document as a list of words:
        word_list = self._extract_word(document=document)

        # Update words indexes:
        self._update_indexes(word_list=word_list)

        # Replace the words by their index:
        words_indexes_list = [self.word_to_index[word] for word in word_list]

        # Sample empirical topic assignment:
        statistics, batch_document_topic_distribution = self._compute_statistics_components(
            words_indexes_list
        )

        # Oline variational inference
        self._update_weights(
            statistics=statistics)

        if self.counter % self.vocab_prune_interval == 0:
            self._prune_vocabulary()

        return batch_document_topic_distribution

    def fit_one(self, document):
        """
        Updates running latent Dirichlet allocation. Splits the words of the document into a list.
        Updates the word indexes. Runs the online variational inference.

        Args:
            document (str): Current document to update the model.

        Returns
            self
        """
        # Updates number of documents:
        self.counter += 1

        # Extracts words of the document as a list of words:
        word_list = self._extract_word(document=document)

        # Update words indexes:
        self._update_indexes(word_list=word_list)

        # Replace the words by their index:
        words_indexes_list = [self.word_to_index[word] for word in word_list]

        # Sample empirical topic assignment:
        statistics, _ = self._compute_statistics_components(
            words_indexes_list
        )

        # Oline variational inference
        self._update_weights(
            statistics=statistics)

        if self.counter % self.vocab_prune_interval == 0:
            self._prune_vocabulary()

        return self

    def transform_one(self, document):
        """
        Splits the words of the document into a list. Updates the word indexes.
        Assign topics to the document.

        Args:
            document (str): Current document to be encoded and reduced in size.

        Returns
            list: Components of the input document.
        """
        # Extracts words of the document as a list of words:
        word_list = self._extract_word(document=document)

        # Update words indexes:
        self._update_indexes(word_list=word_list)

        # Replace the words by their index:
        words_indexes_list = [self.word_to_index[word] for word in word_list]

        # Sample empirical topic assignment:
        _, components = self._compute_statistics_components(
            words_indexes_list
        )

        return components

    @classmethod
    def _extract_word(cls, document):
        '''
        Split the sentence into a list of words.

        Args:
            document (str): Input document.
        Returns:
            list: Words of the input document.
        '''
        return document.split(' ')

    def _update_indexes(self, word_list):
        """
        Adds the words of the document to the index if they are not part of the current vocabulary.
        Updates of the number of distinct words seen.

        Args:
            word_list (list): Content of the document as a list of words.

        Returns:
            None
        """
        for word in word_list:
            if word not in self.word_to_index.keys():
                new_index = len(self.word_to_index) + 1
                self.word_to_index[word] = new_index
                self.index_to_word[new_index] = word
                self.truncation_size_prime += 1

    @classmethod
    def _compute_weights(cls, n_components, nu_1, nu_2):
        """
        Calculates the vocabulary weighting according to the word distribution present in the
        vocabulary. The Psi function is the logarithmic derivative of the gamma function.

        Args:
            n_components (int): Number of topics.
            nu_1 (dict): Weights of the words of the vocabulary.
            nu_2 (dict): Weights of the words of the vocabulary.

        Returns:
            Tuple[dict, dict]: Weights of the words of the current vocabulary.
        """
        exp_weights = {}
        exp_oov_weights = {}

        for topic in range(n_components):

            psi_nu_1 = special.psi(nu_1[topic])
            psi_nu_2 = special.psi(nu_2[topic])

            psi_nu_1_nu_2 = special.psi(nu_1[topic] + nu_2[topic])

            psi_nu_1_nu_2_minus_psi_nu_2 = np.cumsum(
                [psi_nu_2 - psi_nu_1_nu_2], axis=1)

            exp_oov_weights[topic] = np.exp(
                psi_nu_1_nu_2_minus_psi_nu_2[0][-1])

            psi_nu_1_nu_2_minus_psi_nu_2 = ndimage.interpolation.shift(
                input=psi_nu_1_nu_2_minus_psi_nu_2[0],
                shift=1,
                cval=0
            )

            exp_weights[topic] = np.exp(
                psi_nu_1 - psi_nu_1_nu_2 + psi_nu_1_nu_2_minus_psi_nu_2
            )

        return exp_weights, exp_oov_weights

    def _update_weights(self, statistics):
        """
        Learns documents and word representations. Calculates the variational approximation.

        Args:
            statistics (defaultdict): Weights associated to the words.

        Returns:
            None
        """
        reverse_cumulated_phi = {}

        for k in range(self.n_components):

            reverse_cumulated_phi[k] = ndimage.interpolation.shift(
                input=statistics[k],
                shift=-1,
                cval=0
            )

            reverse_cumulated_phi[k] = np.flip(reverse_cumulated_phi[k])
            reverse_cumulated_phi[k] = np.cumsum(reverse_cumulated_phi[k])
            reverse_cumulated_phi[k] = np.flip(reverse_cumulated_phi[k])

        # Epsilon will be between 0 and 1.
        # Epsilon value says how much to weight the information we got from this document.
        self.epsilon = (self.tau + self.counter) ** -self.kappa

        for k in range(self.n_components):

            if self.truncation_size < self.truncation_size_prime:

                difference_truncation = self.truncation_size_prime - self.truncation_size

                self.nu_1[k] = np.append(
                    self.nu_1[k], np.ones(difference_truncation))
                self.nu_2[k] = np.append(
                    self.nu_2[k], np.ones(difference_truncation))

            # Variational Approximation
            self.nu_1[k] += (self.epsilon * (self.number_of_documents *
                                             np.array(statistics[k]) + 1 - self.nu_1[k]))

            self.nu_2[k] += (self.epsilon * (self.alpha_beta + self.number_of_documents *
                                             np.array(reverse_cumulated_phi[k]) - self.nu_2[k]))

        self.truncation_size = self.truncation_size_prime

    def _compute_statistics_components(self, words_indexes_list):
        """
        Extract latent variables from the document and words.

        Args:
            words_indexes_list (list): Ids of the words of the input document.

        Returns:
            Tuple[dict, dict]: Computed statistics over the words. Document reprensetation across
            topics.
        """
        statistics = defaultdict(
            lambda: np.zeros(self.truncation_size_prime))

        exp_weights, exp_oov_weights = self._compute_weights(
            n_components=self.n_components,
            nu_1=self.nu_1,
            nu_2=self.nu_2,
        )

        size_vocab = len(words_indexes_list)

        phi = np.random.random((self.n_components, size_vocab))

        phi = phi / np.sum(phi, axis=0)

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
                temp_phi = np.random.multinomial(1, temp_phi)

                phi[:, word_index] = temp_phi

                phi_sum += temp_phi

                if sample_index >= self.burn_in_sweeps:

                    for k in range(self.n_components):

                        index = words_indexes_list[word_index]

                        statistics[k][index] += temp_phi[k]

        document_topic_distribution = self.alpha_theta + phi_sum

        for k in range(self.n_components):

            statistics[k] /= (self.number_of_samples - self.burn_in_sweeps)

        return statistics, document_topic_distribution

    def _prune_vocabulary(self):
        """
        Reduces the size of the index exceeds the maximum size.

        Args:
            None

        Returns:
            None
        """
        if self.nu_1[0].shape[0] > self.maximum_size_vocabulary:

            for topic in range(self.n_components):
                # Updates words latent variables
                self.nu_1[topic] = self.nu_1[topic][:self.maximum_size_vocabulary]
                self.nu_2[topic] = self.nu_2[topic][:self.maximum_size_vocabulary]

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
