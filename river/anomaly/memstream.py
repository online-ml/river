"""
This is an adaptation of the MemStream algorithm for anomaly detection in data streams.
The code was extracted from https://github.com/Stream-AD/MemStream/
and adapted to fit into the River framework.
The original paper can be found: https://arxiv.org/pdf/2106.03837
"""

from __future__ import annotations

import abc
import warnings

import numpy as np
from sklearn.decomposition import PCA

from river import anomaly, utils


class EncoderType:
    DENOISING_AUTOENCODER = "denoising_autoencoder"  # TODO: implement
    PCA = "pca"
    IB = "information_bottleneck"  # TODO: implement


class ReplaceStrategy:
    FIFO = "FIFO"  # First In, First Out
    LRU = "LRU"  # Least Recently Used
    RANDOM = "RANDOM"  # Random replacement


class MemStream(anomaly.base.AnomalyDetector):
    """MemStream: Memory-Based Streaming Anomaly Detection

    MemStream is an **online anomaly detection framework** designed to process
    high-dimensional data streams with potential concept drift. It consists of
    two main components:

    1. A **feature encoder** that transforms raw inputs into a lower-dimensional
        representation. This encoder can be a **PCA-based projection**, or an alternative
        representation learning method.

    2. A **memory module** that maintains a collection of encoded representations
        of recent **normal** data points. This memory acts as a dynamic model
        of the current data distribution and evolves over time to adapt to drift
        without requiring explicit labels.

    For each incoming data point, MemStream computes an **anomaly score** by
    measuring the similarity between the encoded input and the stored memory. If
    the score indicates that the point is close to existing memory entries
    (i.e., similar to previously seen normal data), it is considered normal and
    may be added to memory. Otherwise, it is flagged as anomalous. The memory
    module uses a replacement policy to adapt to changing trends (concept drift)
    and avoid memory poisoning.

    References
    ----------
    - S.Bhatia, A.Jain, S.Srivastava, K.Kawaguchi, B.Hooi
        "MemStream: Memory-Based Streaming Anomaly Detection"
      https://arxiv.org/pdf/2106.03837


    Parameters
    ----------
    memory_size
        The maximum number of encoded normal data points to store in memory.
    max_threshold
        The maximum anomaly score threshold for accepting a new data point into memory.
    encoder_type
        The type of encoder to use: denoising autoencoder, PCA or IB.
    eps
        A small value to prevent division by zero during normalization.
    replace_strategy
        The memory replacement strategy: FIFO, LRU, or RANDOM.
    grace_period
        The number of initial samples to process before starting anomaly scoring.
    k
        The number of nearest neighbors to consider when computing the anomaly score.
    gamma
        The weighting factor for the score computation.
    """

    def __init__(
        self,
        memory_size=5_000,
        max_threshold=0.1,
        encoder_type=EncoderType.PCA,
        eps=1e-8,
        replace_strategy=ReplaceStrategy.FIFO,
        grace_period=5_000,
        k=5,
        gamma=0.1,
    ):
        self.out_dim, self.memory, self.mem_data = None, None, None
        self.eps = eps
        self.memory_size = memory_size
        self.max_threshold = max_threshold
        self.encoder_type = encoder_type
        self.replace_strategy = replace_strategy
        self.grace_period = grace_period
        self.count = 0
        self.encoder = None
        self.defined_encoder = False
        self.mean = None
        self.std = None
        self.initialized = False
        self.k = k
        self.gamma = gamma
        self.exp = np.array([self.gamma**i for i in np.arange(self.k)])
        self.sum_exp = np.sum(self.exp)

    @abc.abstractmethod
    def __define_memory__(self):
        """Define the memory structure and initialization."""

    @abc.abstractmethod
    def __define_encoder__(self, train_data):
        """Function to define the encoder model, its input
        is a list of training samples and corresponding labels.

        Args:
            train_data: List of tuples (x, y) where x is a data point and y is its label.
        """

    @abc.abstractmethod
    def __encode__(self, x):
        """Encode the input data point."""

    def __update_memory__(self, loss_value, encode_x, x):
        if loss_value <= self.max_threshold:
            if self.replace_strategy == ReplaceStrategy.FIFO:
                self.memory[self.count % self.memory_size] = encode_x
                self.mem_data[self.count % self.memory_size] = x
            elif (
                self.replace_strategy == ReplaceStrategy.LRU
            ):  # If we use LRU we will reorder the memory based on usage
                repl_index = self.count if self.count < self.memory_size else -1
                self.memory[repl_index] = encode_x
                self.mem_data[repl_index] = x
            elif self.replace_strategy == ReplaceStrategy.RANDOM:
                rand_pos = np.random.randint(0, self.memory_size)
                self.memory[rand_pos] = encode_x
                self.mem_data[rand_pos] = x

            self.mean, self.std = self.mem_data.mean(0), self.mem_data.std(0)
            self.count += 1

    def __reorder_memory__(self, memory_index):
        """Reorder the memory to move the accessed memory to the most recently used position.
        This function will be used for the LRU replacement strategy.
        """
        self.memory = np.r_[
            self.memory[memory_index],
            self.memory[:memory_index],
            self.memory[memory_index + 1 : self.count],
            self.memory[self.count :],
        ]
        self.mem_data = np.r_[
            self.mem_data[memory_index],
            self.mem_data[:memory_index],
            self.mem_data[memory_index + 1 : self.count],
            self.mem_data[self.count :],
        ]

    def __format_x__(self, x):
        if not hasattr(self, "_feature_order"):
            self._feature_order = sorted(x.keys())
            self.in_dim = len(self._feature_order)
            self.out_dim, self.memory, self.mem_data = self.__define_memory__()
        if not isinstance(x, np.ndarray):
            x = utils.VectorDict(x)
            x = x.to_numpy(self._feature_order)
        return x

    def __process_x__(self, x):
        """
        This normalizes the input, encodes it, computes the anomaly score,
        and updates the memory.
        Args:
            x: The input data point.
        Returns:
            loss_value: The computed anomaly score.
            encode_x: The encoded representation of the input.
            x: The normalized input data point.
        """
        x = self.__format_x__(x)
        new = (x - self.mean) / (self.std)
        new = new.reshape(1, -1)
        new[:, self.std == 0] = 0
        encode_x = self.__encode__(new)
        norms = np.linalg.norm(self.memory - encode_x, ord=1, axis=1)
        if self.k <= 1:
            loss_value = np.min(norms)
        else:
            loss_values = np.sort(
                norms,
            )[: self.k]
            loss_value = np.sum(loss_values * self.exp) / (
                np.sum(self.exp) + self.eps
            )
        if self.replace_strategy == ReplaceStrategy.LRU:
            memory_indeces = np.argsort(norms)[: self.k]
            (
                self.__reorder_memory__(memory_index)
                for memory_index in memory_indeces
            )
        return loss_value, encode_x, x

    def score_one(self, x, y=None):
        """Compute the anomaly score for a new data point."""
        if self.initialized:
            loss_value, _, _ = self.__process_x__(x)
            return loss_value if self.count >= self.grace_period else 0
        else:
            return 0

    def __manage_non_encoded__(self, x, y):
        """Handle the case when the encoder is not yet defined.
        If the encoder is not defined, we collect samples until we reach the grace period,
        then we define the encoder using the collected samples.
        """
        if not self.initialized:
            if self.count < self.grace_period:
                if (y is not None and y != 1) or y is None:
                    self.__update_memory__(0, np.zeros((1, self.out_dim)), x)
            elif self.count >= self.grace_period:
                self.__define_encoder__(
                    [(self.mem_data[i], 0) for i in range(len(self.mem_data))]
                )
                self.initialized = True

    def learn_one(self, x, y=None):
        x = self.__format_x__(x)
        self.__manage_non_encoded__(x, y)
        if self.initialized:
            loss_value, encode_x, x = self.__process_x__(x)
            if y is not None and y == 1:
                return  # Do not learn from anomalies
            self.__update_memory__(
                0 if self.count < self.grace_period else loss_value, encode_x, x
            )


class MemStreamPCA(MemStream):
    """A PCA-based encoder implementation. The values will be saved in memory until
    the grace period is reached, then PCA will be fitted on the normal samples to define
    the encoder.

    Parameters
    ----------
    memory_size
        The maximum number of encoded normal data points to store in memory.
    max_threshold
        The maximum anomaly score threshold for accepting a new data point into memory.
    eps
        A small value to prevent division by zero while computing the score.
    replace_strategy
        The memory replacement strategy: FIFO, LRU, or RANDOM.
    grace_period
        The number of initial samples to process before starting anomaly scoring.
    n_components
        The number of principal components to keep.

    Examples
    --------
    >>> from river import anomaly
    >>> from river import datasets
    >>> from river import metrics
    >>> from river import preprocessing
    >>> np.random.seed(42)
    >>> model = anomaly.QuantileFilter(
    ...     MemStreamPCA(),
    ...     q=0.9
    ... )
    >>> auc = metrics.ROCAUC()
    >>> for x, y in datasets.CreditCard().take(35_000):
    ...     score = model.score_one(x)
    ...     is_anomaly = model.classify(score)
    ...     model.learn_one(x)
    ...     auc.update(y, is_anomaly)
    >>> auc
    ROCAUC: 90.33%
    """

    def __init__(
        self,
        memory_size=1_000,
        max_threshold=0.1,
        eps=1e-8,
        replace_strategy=ReplaceStrategy.FIFO,
        grace_period=5_000,
        n_components=20,
        k=5,
        gamma=0.1,
    ):
        self.n_components = n_components
        super().__init__(
            memory_size=memory_size,
            max_threshold=max_threshold,
            encoder_type=EncoderType.PCA,
            eps=eps,
            replace_strategy=replace_strategy,
            grace_period=grace_period,
            k=k,
            gamma=gamma,
        )
        self.components_ = None
        self.mean_ = None

    def __define_memory__(self):
        memory = np.zeros((self.memory_size, self.n_components))
        mem_data = np.zeros((self.memory_size, self.in_dim))
        return self.n_components, memory, mem_data

    def __define_encoder__(self, train_data):
        self.defined_encoder = True
        x_train, y_train = zip(*train_data)
        x_train = np.array([self.__format_x__(x) for x in x_train])
        y_train = np.array([y for y in y_train])
        num_components = min(
            min(x_train.shape[0], x_train.shape[1]),
            self.n_components,  # to avoid n_components > n_samples
        )
        if num_components < self.n_components:
            warnings.warn(
                f"Number of components was set to {self.n_components}, but there are only "
                f"{x_train.shape[0]} samples and {x_train.shape[1]} features."
                f"Setting number of components to {num_components}."
            )
        self.n_components = num_components
        self.pca = PCA(n_components=self.n_components)
        self.mean, self.std = x_train.mean(0), x_train.std(0)
        new = (x_train - self.mean) / self.std
        new[:, self.std == 0] = 0
        self.pca.fit(np.nan_to_num(new, nan=0.0))
        for elem in x_train[y_train == 0]:  # Fill memory with normal samples
            elem = elem.reshape(1, -1)
            encoded_elem = self.__encode__(elem)
            self.__update_memory__(0, encoded_elem, elem)

    def __encode__(self, x):
        if self.defined_encoder:
            new = (x - self.mean) / (self.std)
            new = new.reshape(1, -1)
            new[:, self.std == 0] = 0
            if len(new.shape) <= 1:
                new = new.reshape(1, -1)
            encoder_output = self.pca.transform(np.nan_to_num(new, nan=0.0))
            return np.array(encoder_output)
        else:
            return np.zeros((1, self.n_components))
