"""
This is an adaptation of the MemStream algorithm for anomaly detection in data streams.
The code was extracted form https://github.com/Stream-AD/MemStream/
and adapted to fit into the River framework.
The original paper can be found: https://arxiv.org/pdf/2106.03837
"""

from __future__ import annotations

import abc
import warnings

import numpy as np
from sklearn import metrics
from sklearn.decomposition import PCA
from tqdm import tqdm  # type: ignore[import-untyped]

from river import anomaly, utils
from river.optim import Adam


class EncoderType:
    DENOISING_AUTOENCODER = "denoising_autoencoder"
    PCA = "pca"
    IB = "information_bottleneck"


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
        representation. This encoder can be a **denoising autoencoder**, **PCA-based
        projection**, or an alternative representation learning method.
        The goal is to capture the structure of the normal data distribution. :contentReference[oaicite:1]{index=1}

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
    """

    def __init__(
        self,
        memory_size=1000,
        max_threshold=0.5,
        encoder_type=EncoderType.DENOISING_AUTOENCODER,
        eps=1e-8,
        replace_strategy=ReplaceStrategy.FIFO,
        grace_period=100,
    ):
        self.out_dim, self.memory, self.mem_data = None, None, None
        self.eps = eps
        self.memory_size = memory_size
        self.max_threshold = max_threshold
        self.encoder_type = encoder_type
        self.replace_strategy = replace_strategy
        self.grace_period = grace_period
        self.sample_count = 0
        self.clock = 0
        self.count = 0
        self.encoder = None
        self.defined_encoder = False
        self.mean = None
        self.std = None
        self.initialized = False

    @abc.abstractmethod
    def define_memory(self):
        """Define the memory structure and initialization."""

    def learn_many(self, x_train, y_train):
        """Function to learn multiple data points at once.
        The input should be a list of numpy arrays.

        Args:
            x_train: List of data points to learn.
            y_train: List of corresponding labels.
        """
        for elem in range(len(y_train)):
            if y_train[elem] == 0:
                self.learn_one(x_train[elem])

    @abc.abstractmethod
    def define_encoder(self, train_data):
        """Function to define the encoder model, its input
        is a list of training samples and corresponding labels.

        Args:
            train_data: List of tuples (x, y) where x is a data point and y is its label.
        """

    @abc.abstractmethod
    def __encode__(self, x):
        """Encode the input data point."""

    def update_memory(self, loss_value, encode_x, x):
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
            return 1
        return 0

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

    # TODO: Remove this function
    def __format_x__(self, x):
        if not hasattr(self, "_feature_order"):
            self._feature_order = sorted(x.keys())
            self.in_dim = len(self._feature_order)
            self.out_dim, self.memory, self.mem_data = self.define_memory()
        if not isinstance(x, np.ndarray):
            x = utils.VectorDict(x)
            x = x.to_numpy(self._feature_order)
        return x

    def __process_x__(self, x):
        x = self.__format_x__(x)
        new = (x - self.mean) / (self.std + self.eps)
        new[self.std == 0] = 0
        encode_x = self.__encode__(new)
        norms = np.linalg.norm(self.memory - encode_x, ord=1, axis=1)
        loss_value = np.min(norms)
        if self.replace_strategy == ReplaceStrategy.LRU:
            memory_index = np.argmin(norms)
            self.__reorder_memory__(memory_index)
        return loss_value, encode_x, x

    def score_one(self, x, y=None):
        """Compute the anomaly score for a new data point."""
        if self.initialized:
            # defined_encoder = self.__manage_non_encoded__(x, y)
            if self.defined_encoder:
                loss_value, _, _ = self.__process_x__(x)
                return loss_value if self.count >= self.grace_period else 0
            else:
                return 0
        else:
            return 0

    def __manage_non_encoded__(self, x, y):
        """Handle the case when the encoder is not yet defined.
        If the encoder is not defined, we collect samples until we reach the grace period,
        then we define the encoder using the collected samples.
        """
        if not self.defined_encoder:
            if self.count < self.grace_period:
                self.count += 1
                if y is not None and y != 1:
                    self.update_memory(0, np.zeros((1, self.out_dim)), x)
                warnings.warn(
                    "Encoder not defined. Call define_encoder first.",
                    RuntimeWarning,
                )
                return False
            elif self.count >= self.grace_period:
                print(
                    "Grace period ended",
                    "Grace Period: ",
                    self.count,
                    "Memory Size: ",
                    len(self.mem_data),
                )
                self.define_encoder(
                    [(self.mem_data[i], 0) for i in range(len(self.mem_data))]
                )
                return True
        else:
            return True

    def learn_one(self, x, y=None):
        x = self.__format_x__(x)
        if not self.initialized:
            self.initialized = True
        encoder_defined = self.__manage_non_encoded__(x, y)
        if encoder_defined:
            loss_value, encode_x, x = self.__process_x__(x)
            if y is not None and y == 1:
                return  # Do not learn from anomalies
            self.update_memory(
                0 if self.count < self.grace_period else loss_value, encode_x, x
            )


class MemStreamAutoencoder(MemStream):
    """The denoising autoencoder implementation using numpy.

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
    learning_rate
        The learning rate for training the autoencoder.
    epochs
        The number of epochs to train the autoencoder.
    batch_size
        The batch size for training the autoencoder.

    Examples
    --------
    >>> from river import anomaly
    >>> from river import datasets
    >>> from river import metrics
    >>> from river import preprocessing
    >>> np.random.seed(42)
    >>> model = anomaly.QuantileFilter(
    ...     MemStreamAutoencoder(memory_size=1000, max_threshold=80),
    ...     q=0.9
    ... )
    >>> auc = metrics.ROCAUC()
    >>> model.anomaly_detector.define_encoder(datasets.CreditCard().take(2_000))
    >>> for x, y in datasets.CreditCard().take(15_000):
    ...     score = model.score_one(x)
    ...     is_anomaly = model.classify(score)
    ...     model.learn_one(x)
    ...     auc.update(y, is_anomaly)
    >>> auc
    ROCAUC: 93.27%

    """

    def __init__(
        self,
        memory_size=1000,
        max_threshold=0.5,
        eps=1e-8,
        replace_strategy=ReplaceStrategy.FIFO,
        grace_period=100,
        learning_rate=0.03,
        epochs=100,
        batch_size=32,
    ):
        super().__init__(
            memory_size=memory_size,
            max_threshold=max_threshold,
            encoder_type=EncoderType.DENOISING_AUTOENCODER,
            eps=eps,
            replace_strategy=replace_strategy,
            grace_period=grace_period,
        )

        self.learning_rate = learning_rate
        self.optimizer = Adam(lr=self.learning_rate)
        self.epochs = epochs
        self.batch_size = batch_size

    def define_memory(self):
        out_dim = self.in_dim * 2
        memory = np.zeros((self.memory_size, out_dim))
        mem_data = np.zeros((self.memory_size, self.in_dim))
        return out_dim, memory, mem_data

    def define_encoder(self, train_data):
        if not self.initialized:
            self.initialized = True
        x_train, y_train = zip(*train_data)
        x_train = np.array([self.__format_x__(x) for x in x_train])
        y_train = np.array([y for y in y_train])
        self.mean = np.mean(x_train, axis=0)
        self.std = np.std(x_train, axis=0)
        pbar = tqdm(range(self.epochs), desc="Training Autoencoder")
        self.W1 = np.random.randn(self.in_dim, self.out_dim) * 0.01  # encoder
        self.b1 = np.zeros((1, self.out_dim))  # encoder
        self.W2 = np.random.randn(self.out_dim, self.in_dim) * 0.01  # decoder
        self.b2 = np.zeros((1, self.in_dim))  # decoder
        self.params = {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
        }
        for epoch in pbar:
            # use the batch size to train
            for i in range(0, len(y_train), self.batch_size):
                x_batch = x_train[i : i + self.batch_size]
                y_batch = y_train[i : i + self.batch_size]
                x_batch = x_batch - self.mean
                x_batch = x_batch / (self.std + self.eps)
                # x_batch[:, self.std == 0] = 0
                output = self.__forward__(x_batch)
                batch_loss = metrics.mean_squared_error(x_batch, output)
                self.__backward__(x_batch, output)
                (
                    self.update_memory(0, self.__encode__(x), x)
                    for x, y in zip(x_batch, y_batch)
                    if y != 1
                )

            pbar.set_postfix({"MSE Loss": f"{batch_loss:.6f}"})
        self.defined_encoder = True

    def __encode__(self, x):
        # defined_encoder = self.__manage_non_encoded__(x, 0)
        if self.defined_encoder:
            return x @ self.W1 + self.b1
        else:
            return np.zeros((1, self.out_dim))

    def __forward__(self, x):
        # Encoder
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)
        # Decoder
        z2 = np.dot(self.a1, self.W2) + self.b2
        output = z2
        return output

    def __backward__(self, x, output):
        """Backpropagation to update weights. The loss used will be MSE.
        Args:
            x (np.ndarray): Input data.
            output (np.ndarray): Reconstructed output from the autoencoder.
        """
        m = x.shape[0] if len(x.shape) > 1 else 1
        dz2 = output - x
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (1 - np.tanh(self.z1) ** 2)
        dW1 = np.dot(x.T.reshape(-1, m), dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        grads = {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
        }
        self.params = self.optimizer._step_with_dict(self.params, grads)
        self.__update_parameters__()

    def __update_parameters__(self):
        self.W1 = self.params["W1"]
        self.b1 = self.params["b1"]
        self.W2 = self.params["W2"]
        self.b2 = self.params["b2"]


class MemStreamPCA(MemStream):
    """A simple PCA-based encoder implementation.

    Parameters
    ----------
    memory_size
        The maximum number of encoded normal data points to store in memory.
    max_threshold
        The maximum anomaly score threshold for accepting a new data point into memory.
    eps
        A small value to prevent division by zero during normalization.
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
    ...     MemStreamPCA(memory_size=1000, max_threshold=80),
    ...     q=0.9
    ... )
    >>> auc = metrics.ROCAUC()
    >>> model.anomaly_detector.define_encoder(datasets.CreditCard().take(2_000))
    >>> for x, y in datasets.CreditCard().take(15_000):
    ...     score = model.score_one(x)
    ...     is_anomaly = model.classify(score)
    ...     model.learn_one(x)
    ...     auc.update(y, is_anomaly)
    >>> auc
    ROCAUC: 73.44%
    """

    def __init__(
        self,
        memory_size=1000,
        max_threshold=5,
        eps=1e-8,
        replace_strategy=ReplaceStrategy.FIFO,
        grace_period=100,
        n_components=2,
    ):
        self.n_components = n_components
        super().__init__(
            memory_size=memory_size,
            max_threshold=max_threshold,
            encoder_type=EncoderType.PCA,
            eps=eps,
            replace_strategy=replace_strategy,
            grace_period=grace_period,
        )
        self.components_ = None
        self.mean_ = None

    def define_memory(self):
        out_dim = self.n_components
        memory = np.zeros((self.memory_size, out_dim))
        mem_data = np.zeros((self.memory_size, self.in_dim))
        return out_dim, memory, mem_data

    def define_encoder(self, train_data):
        if not self.initialized:
            self.initialized = True
        self.defined_encoder = True
        x_train, y_train = zip(*train_data)
        x_train = np.array([self.__format_x__(x) for x in x_train])
        y_train = np.array([y for y in y_train])
        self.n_components = min(
            min(x_train.shape[0], x_train.shape[1]), self.n_components
        )
        self.pca = PCA(n_components=self.n_components)
        self.mean, self.std = x_train.mean(0), x_train.std(0)
        new = (x_train - self.mean) / self.std
        # new[:, self.std == 0] = 0
        self.pca.fit(np.nan_to_num(new, nan=0.0))
        for elem in x_train[y_train == 0]:
            elem = elem.reshape(1, -1)
            encoded_elem = self.__encode__(elem)
            self.update_memory(0, encoded_elem, elem)

    def __encode__(self, x):
        # defined_encoder = self.__manage_non_encoded__(x, 0)
        if self.defined_encoder:
            new = (x - self.mean) / (self.std + self.eps)
            if len(new.shape) <= 1:
                new = new.reshape(1, -1)
            encoder_output = self.pca.transform(np.nan_to_num(new, nan=0.0))
            return np.array(encoder_output)
        else:
            return np.zeros((1, self.n_components))
