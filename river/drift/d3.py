from copy import deepcopy

from river import metrics
from river.base import DriftDetector
from river.tree import HoeffdingTreeClassifier


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
    window_size
        The size of the data window. (int)
    auc_threshold
        Required AUC score to signal a drift.
        From 0.5 to 1.0


    Examples
    --------
    >>> from river import synth
    >>> from river.drift import D3

    >>> d3 = D3(seed=12345)

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

    _AUC_NUM_THRESHOLDS = 20
    _LABEL_FOR_NEW_DATA = True
    _LABEL_FOR_OLD_DATA = False

    def __init__(
        self,
        window_size=200,
        auc_threshold=0.7,
        discriminative_classifier=HoeffdingTreeClassifier(grace_period=40, max_depth=3),
    ):
        super().__init__()
        self.auc_threshold = auc_threshold
        self.sub_window_size = int(window_size / 2)
        self.discriminative_classifier = discriminative_classifier
        self.old_data_window = [None] * self.sub_window_size
        self.new_data_window = [None] * self.sub_window_size
        self.data_labels = None
        self.store_labels = False
        self.old_data_window_index = 0
        self.new_data_window_index = 0
        self.auc = metrics.ROCAUC(n_thresholds=D3._AUC_NUM_THRESHOLDS)

        super().reset()

    def current_data_and_labels(self):
        """Returns the data and labels for the current data window.

        They can be used when for retraining the stream classifier after drift detection.
        """
        return self.old_data_window, self.data_labels

    def update_labels_if_storing_enabled(self, index, label):
        """Update the labels array if storing is enabled"""
        if not self.store_labels:
            return
        self.data_labels[index] = label

    def reset(self):
        """Reset the change detector."""
        super().reset()
        self.old_data_window = [None] * self.sub_window_size
        self.new_data_window = [None] * self.sub_window_size
        self.old_data_window_index = 0
        self.new_data_window_index = 0
        self.data_labels = None
        self.store_labels = False
        self.auc = metrics.ROCAUC(n_thresholds=D3._AUC_NUM_THRESHOLDS)
        self.discriminative_classifier = self.discriminative_classifier.clone()

    def update(self, sample, label=None):
        """Update the change detector with a single sample.

        Parameters
        ----------
        sample
            An instance from the data stream (dict) having N items (number of features).
        label
            Class label for sample.

        Notes
        -----
        * The label is is not used in the detection process. It can be set to None.
        * If the label is not set to None, D3 will store the last window_size/2
          class labels of the samples. It is useful for retraining the stream classifier
          when a drift is detected.
        """
        if self._in_concept_change:
            self._in_concept_change = False

        # Start storing labels if not None
        if (label is not None) and (self.store_labels is False):
            self.data_labels = [None] * self.sub_window_size
            self.store_labels = True

        if self.old_data_window_index < self.sub_window_size:
            self.old_data_window[self.old_data_window_index] = sample
            self.old_data_window_index += 1
            return self._in_concept_change, self._in_warning_zone

        self.new_data_window[self.new_data_window_index] = sample
        self.update_labels_if_storing_enabled(self.new_data_window_index, label)

        # Updating discriminative classifier with a sample from the old and new data
        old_data_sample = self.old_data_window[self.new_data_window_index]
        self.discriminative_classifier.learn_one(sample, D3._LABEL_FOR_NEW_DATA)
        self.discriminative_classifier.learn_one(
            old_data_sample, D3._LABEL_FOR_OLD_DATA
        )

        # Update AUC
        prob_new = self.discriminative_classifier.predict_proba_one(sample)[1]
        prob_old = self.discriminative_classifier.predict_proba_one(old_data_sample)[1]
        self.auc = self.auc.update(D3._LABEL_FOR_NEW_DATA, prob_new)
        self.auc = self.auc.update(D3._LABEL_FOR_OLD_DATA, prob_old)

        self.new_data_window_index += 1

        if self.new_data_window_index == self.sub_window_size:
            auc_score = self.auc.get()
            if (auc_score > self.auc_threshold) or (
                auc_score < self.auc_threshold - 0.5
            ):
                self._in_concept_change = True
            self.old_data_window = deepcopy(self.new_data_window)
            self.new_data_window_index = 0
            self.auc = metrics.ROCAUC(n_thresholds=D3._AUC_NUM_THRESHOLDS)
            self.discriminative_classifier = self.discriminative_classifier.clone()

        return self._in_concept_change, self._in_warning_zone
