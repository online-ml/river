import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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
        The size of the old data window.
    new_data_fraction
        Determines the size of the new data window. It is calculated as
        a fraction of the old data window. The new data window has size
        new data fraction * old data window size.
    auc_threshold
        Required AUC score to signal a drift.
    discriminative_classifier
        Classifier to be used to distinguish old data from the new. If it
        is set to None, by default Logistic Regression with default
        parameters will be used.

    Examples
    --------
    >>> import numpy as np
    >>> from river.drift import D3
    >>> np.random.seed(12345)

    >>> d3 = D3()

    >>> # Simulate a data stream as a normal distribution of 1's and 0's
    >>> data_stream = np.random.randint(2, size=2000)
    >>> # Change the data distribution from index 999 to 1500, simulating an
    >>> # increase in error rate (1 indicates error)
    >>> data_stream[999:1500] = 1

    >>> # Update drift detector and verify if change is detected
    >>> for i, val in enumerate(data_stream):
    ...     in_drift, in_warning = ddm.update(val)
    ...     if in_drift:
    ...         print(f"Change detected at index {i}, input value: {val}")
    Change detected at index 1077, input value: 1

    References
    ----------
    [^1]: Ömer Gözüaçık, Alican Büyükçakır, Hamed Bonab, Fazli Can: Unsupervised concept drift detection with a discriminative classifier. CIKM 2019: 2365-2368

    """

    def __init__(
        self,
        old_data_window_size=250,
        new_data_fraction=0.3,
        auc_threshold=0.7,
        discriminative_classifier=None,
    ):
        super().__init__()
        self.old_data_window_size = old_data_window_size
        self.new_data_fraction = new_data_fraction
        self.auc_threshold = auc_threshold
        self.discriminative_classifier = discriminative_classifier
        self.new_data_window_size = int(
            self.old_data_window_size * self.new_data_fraction
        )
        self.full_window_size = self.old_data_window_size + self.new_data_window_size
        self.data_sliding_window = None
        self.sliding_window_index = 0
        super().reset()

    def _is_sliding_window_full(self):
        """Checks if the sliding window is full."""
        if self.sliding_window_index < self.full_window_size:
            return False
        return True

    def _discriminative_drift_detection(self, old_data, new_data):
        """Check if old and new data are seperable"""
        slack_labels = np.concatenate(
            (np.zeros(self.new_data_window_size), np.ones(self.old_data_window_size)),
            axis=0,
        )
        combined_data = np.concatenate((new_data, old_data), axis=0)
        if self.discriminative_classifier is None:
            self.discriminative_classifier = LogisticRegression(solver="liblinear")
        predictions = np.zeros(slack_labels.shape)
        skf = StratifiedKFold(n_splits=2, shuffle=True)
        for train_index, test_index in skf.split(combined_data, slack_labels):
            X_train, X_test = combined_data[train_index], combined_data[test_index]
            y_train = slack_labels[train_index]
            self.discriminative_classifier.fit(X_train, y_train)
            probs = self.discriminative_classifier.predict_proba(X_test)[:, 1]
            predictions[test_index] = probs
        auc_score = roc_auc_score(slack_labels, predictions)
        if auc_score > self.auc_threshold:
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
            An instance (N,1) from the data stream where N is the number of features.

        """
        if self._in_concept_change:
            self._in_concept_change = False

        if self.data_sliding_window is None:
            self.data_sliding_window = np.zeros((self.full_window_size, len(sample)))

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
