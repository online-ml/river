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
        representation. This encoder is a **PCA-based projection**. A PCA-based encoder
        implementation. The values will be saved in memory until the grace period is
        reached, then PCA will be fitted on the normal samples to define the encoder.

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
    rpl_stg
        The memory replacement strategy: FIFO, LRU, or RANDOM.
    grace_period
        The number of initial samples to process before starting anomaly scoring.
    k
        The number of nearest neighbors to consider when computing the anomaly score.
    gamma
        The weighting factor for the score computation.
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
    ...     MemStream(),
    ...     q=0.9
    ... )
    >>> auc = metrics.ROCAUC()
    >>> for x, y in datasets.CreditCard().take(50_000):
    ...     score = model.score_one(x)
    ...     is_anomaly = model.classify(score)
    ...     model.learn_one(x)
    ...     auc.update(y, is_anomaly)
    >>> auc
    ROCAUC: 87.96%
    """

    def __init__(
        self,
        memory_size=1_000,
        max_threshold=10,
        rpl_stg=ReplaceStrategy.FIFO,
        grace_period=1_000,
        k=5,
        gamma=0.25,
        n_components=20,
    ):
        self.out_dim, self.memory, self.mem_data = None, None, None
        self.memory_size = memory_size
        self.max_threshold = max_threshold
        self.rpl_stg = rpl_stg
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
        self.n_components = n_components

    def _define_memory(self):
        """Define the memory structure and initialization."""
        memory = np.zeros((self.memory_size, self.n_components))
        mem_data = np.zeros((self.memory_size, self.in_dim))
        return self.n_components, memory, mem_data

    def _normalize(self, x):
        """Normalize the input data point."""
        new = (x - self.mean) / self.std
        new = new.reshape(1, -1)
        new[:, self.std == 0] = 0
        return new

    def _define_encoder(self, train_data):
        """Function to define the encoder model, its input
        is a list of training samples and corresponding labels.

        Args:
            train_data: List of tuples (x, y) where x is a data point and y is its label.
        """
        self.defined_encoder = True
        x_train, y_train = zip(*train_data)
        x_train = np.array([self._format_x(x) for x in x_train])
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
            new = self._normalize(elem)
            encoded_elem = self._encode(new)
            self._update_memory(0, encoded_elem, elem)

    def _encode(self, x):
        """Encode the input data point."""
        if self.defined_encoder:
            encoder_output = self.pca.transform(np.nan_to_num(x, nan=0.0))
            return np.array(encoder_output)
        else:
            return np.zeros((1, self.n_components))

    def _update_memory(self, loss_value, encode_x, x):
        if loss_value > self.max_threshold:
            return

        idx_rpl = (
            np.random.randint(0, self.memory_size)
            if self.rpl_stg == ReplaceStrategy.RANDOM
            else (
                self.count % self.memory_size
                if self.rpl_stg == ReplaceStrategy.FIFO
                else len(self.memory) - 1
            )
        )
        self._index_replace(self.memory, idx_rpl, encode_x)
        self._index_replace(self.mem_data, idx_rpl, x)

        if self.rpl_stg == ReplaceStrategy.LRU:
            self._move_to_front(self.memory, idx_rpl)
            self._move_to_front(self.mem_data, idx_rpl)

        self.mean = self.mem_data.mean(0)
        self.std = self.mem_data.std(0)
        self.count += 1

    def _index_replace(self, arr, index, value):
        """Replace the value at the specified index in the array."""
        arr[index] = value

    def _move_to_front(self, arr, idx):
        """Reorder the memory to move the accessed memory to the most recently used position.
        This function will be used for the LRU replacement strategy.
        """
        temp = arr[idx].copy()
        arr[1 : idx + 1] = arr[:idx]
        arr[0] = temp

    def _format_x(self, x):
        if not isinstance(x, np.ndarray):
            x = utils.VectorDict(x)
            x = x.to_numpy(self._feature_order)
        return x

    def _process_x(self, x):
        """
        This normalizes the input, encodes it, computes the anomaly score,
        and updates the memory.
        Args:
            x: The input data point.
        Returns:
            loss_value: The computed anomaly score.
            encode_x: The encoded representation of the input.
            x: The input data point.
        """
        x = self._format_x(x)
        new = self._normalize(x)
        encode_x = self._encode(new)
        norms = np.linalg.norm(self.memory - encode_x, ord=1, axis=1)
        if self.k <= 1:
            loss_value = np.min(norms)
        else:
            loss_values = np.sort(
                norms,
            )[: self.k]
            loss_value = np.sum(loss_values * self.exp) / (np.sum(self.exp))
        if self.rpl_stg == ReplaceStrategy.LRU:
            memory_indeces = np.argsort(norms)[: self.k]
            for index in memory_indeces:
                self._move_to_front(self.memory, index)
                self._move_to_front(self.mem_data, index)
        return loss_value, encode_x, x

    def score_one(self, x, y=None):
        """Compute the anomaly score for a new data point."""
        if self.initialized and hasattr(self, "_feature_order"):
            loss_value, _, _ = self._process_x(x)
            return loss_value if self.count >= self.grace_period else 0
        return 0

    def _manage_non_encoded(self, x, y):
        """Handle the case when the encoder is not yet defined.
        If the encoder is not defined, we collect samples until we reach the grace period,
        then we define the encoder using the collected samples.
        """
        if not self.initialized:
            if self.count < self.grace_period:
                if (y is not None and y != 1) or y is None:
                    self._update_memory(0, np.zeros((1, self.out_dim)), x)
            elif self.count >= self.grace_period:
                self._define_encoder(
                    list(zip(self.mem_data, [0] * len((self.mem_data))))
                )
                self.initialized = True

    def learn_one(self, x, y=None):
        if not hasattr(self, "_feature_order"):
            self._feature_order = sorted(x.keys())
            self.in_dim = len(self._feature_order)
            self.out_dim, self.memory, self.mem_data = self._define_memory()
        x = self._format_x(x)
        self._manage_non_encoded(x, y)
        if self.initialized:
            loss_value, encode_x, x = self._process_x(x)
            if y is not None and y == 1:
                return  # Do not learn from anomalies
            self._update_memory(
                0 if self.count < self.grace_period else loss_value, encode_x, x
            )
