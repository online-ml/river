import numpy as np
from sklearn.linear_model import LogisticRegression

from river import metrics
from river.base import DriftDetector


class D3(DriftDetector):
    r"""Drift Detection Method

    D3 (Discriminative Drift Detector) is an unsupervised drift detection
    method which uses a discriminative classifier that can be used with any
    online algorithm without a built-in drift detector. It holds a fixed
    size sliding window of the latest data having two sets: the old and the
    new. A simple classifier is trained to distinguish these sets. It
    detects a drift with respect to classifier performance (AUC).

    Parameters
    ----------
    old_data_window_size
        The size of the old data window. (int)
    new_data_percentage
        Determines the size of the new data window. It is calculated as
        a fraction of the old data window. The new data window has size
        new data percentage * old data window size. (float)
        From 0.0 to 1.0
    auc_threshold
        Required AUC score to signal a drift. (float)
        From 0.5 to 1.0
    discriminative_classifier
        Classifier to be used to distinguish old data from the new data.
        If None, Logistic Regression with default parameters will be used.
        It is advised to use a simple model as the goal of this classifier
        is to determine if the old data and the new data are seperable,
        not to classify them. (sklearn classifier)
    seed
        Is used as as random state for numpy


    Examples
    --------
    >>> from river import synth
    >>> from river.drift import D3
    >>> np.random.seed(12345)

    >>> d3 = D3()

    >>> # Simulate a data stream
    >>> data_stream = synth.Hyperplane(seed=42, n_features=10, mag_change=0.5)

    >>> # Update drift detector and verify if change is detected
    >>> i = 0
    >>> for x, y in data_stream.take(250):
    ...     in_drift, in_warning = d3.update(x)
    ...     if in_drift:
    ...         print(f"Change detected at index {i}")
    ...     i += 1
    Change detected at index 242

    References
    ----------
    [^1]: Ömer Gözüaçık, Alican Büyükçakır, Hamed Bonab, Fazli Can: Unsupervised concept drift detection with a discriminative classifier. CIKM 2019: 2365-2368

    """

    _DISC_CLF_TRAINING_DATA_PERCENTAGE = 0.7

    def __init__(
        self,
        old_data_window_size=100,
        new_data_percentage=0.1,
        auc_threshold=0.7,
        discriminative_classifier=None,
        seed=None,
    ):
        super().__init__()
        self.auc_threshold = auc_threshold
        self.data_sliding_window = None
        self.discriminative_classifier = discriminative_classifier
        self.new_data_percentage = new_data_percentage
        self.feature_names = None
        self.old_data_window_size = old_data_window_size
        self.sliding_window_index = 0
        self.new_data_window_size = int(
            self.old_data_window_size * self.new_data_percentage
        )
        self.full_window_size = self.old_data_window_size + self.new_data_window_size
        np.random.seed(seed)
        super().reset()

    def _is_sliding_window_full(self):
        """Checks if the sliding window is full."""
        if self.sliding_window_index < self.full_window_size:
            return False
        return True

    def _discriminative_drift_detection(self, old_data, new_data):
        """Check if old and new data are seperable"""
        new_train_data_size = int(
            D3._DISC_CLF_TRAINING_DATA_PERCENTAGE * self.new_data_window_size
        )
        old_train_data_size = int(
            D3._DISC_CLF_TRAINING_DATA_PERCENTAGE * self.old_data_window_size
        )
        new_data_train_idx = np.random.randint(
            self.new_data_window_size, size=new_train_data_size
        )
        old_data_train_idx = np.random.randint(
            self.old_data_window_size, size=old_train_data_size
        )
        new_data_labels = np.zeros(self.new_data_window_size)
        old_data_labels = np.ones(self.old_data_window_size)
        train_data = np.concatenate(
            (new_data[new_data_train_idx], old_data[old_data_train_idx]), axis=0
        )
        test_data = np.concatenate(
            (new_data[~new_data_train_idx], old_data[~old_data_train_idx]), axis=0
        )
        train_labels = np.concatenate(
            (new_data_labels[new_data_train_idx], old_data_labels[old_data_train_idx]),
            axis=0,
        )
        test_labels = np.concatenate(
            (
                new_data_labels[~new_data_train_idx],
                old_data_labels[~old_data_train_idx],
            ),
            axis=0,
        )
        if self.discriminative_classifier is None:
            self.discriminative_classifier = LogisticRegression(solver="liblinear")
        self.discriminative_classifier.fit(train_data, train_labels)
        predictions = self.discriminative_classifier.predict_proba(test_data)[:, 1]
        auc = metrics.ROCAUC()
        for yt, yp in zip(test_labels, predictions):
            auc.update(bool(yt), yp)
        auc_score = auc.get()
        if (auc_score > self.auc_threshold) or (auc_score < self.auc_threshold - 0.5):
            return True
        return False

    def reset(self):
        """Reset the change detector."""
        super().reset()
        self.data_sliding_window = None
        self.sliding_window_index = 0

    def update(self, sample):
        """Update the change detector with a single sample.

        Parameters
        ----------
        sample
            An instance from the data stream (dict) having N items (number of features).

        """
        if self._in_concept_change:
            self._in_concept_change = False

        if not self.feature_names:
            self.feature_names = sample.keys()

        sample = [sample[feature] for feature in sorted(self.feature_names)]

        if self.data_sliding_window is None:
            self.data_sliding_window = np.zeros(
                (self.full_window_size, len(self.feature_names))
            )

        if not self._is_sliding_window_full():
            self.data_sliding_window[self.sliding_window_index] = sample
            self.sliding_window_index += 1
            return self._in_concept_change, self._in_warning_zone

        old_data = self.data_sliding_window[: self.old_data_window_size]
        new_data = self.data_sliding_window[self.old_data_window_size :]

        if self._discriminative_drift_detection(old_data, new_data):
            self.sliding_window_index = self.new_data_window_size
            self.data_sliding_window = np.roll(
                self.data_sliding_window, -1 * self.old_data_window_size, axis=0
            )
            self._in_concept_change = True
        else:
            self.sliding_window_index = self.old_data_window_size
            self.data_sliding_window = np.roll(
                self.data_sliding_window, -1 * self.new_data_window_size, axis=0
            )

        return self._in_concept_change, self._in_warning_zone
