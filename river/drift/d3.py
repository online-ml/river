import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

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
        Is used as as random state for StratifiedKFold. If None, a randomly
        generated value by StratifiedKFold will be used while shuffling
        data.


    Examples
    --------
    >>> from river import synth
    >>> from river.drift import D3

    >>> d3 = D3(seed=12345)

    >>> # Simulate a data stream
    >>> data_stream = synth.Hyperplane(seed=42, n_features=10, mag_change=0.5)

    >>> # Update drift detector and verify if change is detected
    >>> i = 0
    >>> for x, y in data_stream.take(500):
    ...     in_drift, in_warning = d3.update(list(x.values()))
    ...     if in_drift:
    ...         print(f"Change detected at index {i}")
    ...     i += 1
    Change detected at index 352

    References
    ----------
    [^1]: Ömer Gözüaçık, Alican Büyükçakır, Hamed Bonab, Fazli Can: Unsupervised concept drift detection with a discriminative classifier. CIKM 2019: 2365-2368

    """

    def __init__(
        self,
        old_data_window_size=100,
        new_data_percentage=0.1,
        auc_threshold=0.7,
        discriminative_classifier=None,
        seed=None,
    ):
        super().__init__()
        self.old_data_window_size = old_data_window_size
        self.new_data_percentage = new_data_percentage
        self.auc_threshold = auc_threshold
        self.discriminative_classifier = discriminative_classifier
        self.seed = seed
        self.new_data_window_size = int(
            self.old_data_window_size * self.new_data_percentage
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
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=self.seed)
        for train_index, test_index in skf.split(combined_data, slack_labels):
            X_train, X_test = combined_data[train_index], combined_data[test_index]
            y_train = slack_labels[train_index]
            self.discriminative_classifier.fit(X_train, y_train)
            probs = self.discriminative_classifier.predict_proba(X_test)[:, 1]
            predictions[test_index] = probs
        auc_score = roc_auc_score(slack_labels, predictions)
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
