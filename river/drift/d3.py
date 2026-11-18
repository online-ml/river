from __future__ import annotations

import typing

from river import base, metrics
from river.base import DriftDetector


class D3(DriftDetector):
    """D3: a discriminative drift detector (Density Ratio-based Detection of Drift).

    D3 keeps two rolling windows of recent feature vectors, an old window
    and a new window, and trains a binary classifier to tell them apart.
    While the two windows come from the same concept, the classifier can do
    no better than chance, and the running ROC AUC stays high; when the
    concept changes, the classifier separates the windows and the AUC drops.
    Drift is signaled when the AUC falls below a threshold (or when it is
    indistinguishable from chance-level performance). After an alarm the
    windows rotate, so the new concept immediately becomes the reference
    context without an unmonitored gap.

    The detector is multivariate by nature: ``update`` expects a dict of
    features. It is the first dict-input detector in ``river.drift``.

    Parameters
    ----------
    discriminative_classifier
        A binary classifier with ``learn_one`` / ``predict_proba_one``,
        trained internally to discriminate the old window from the new
        window. A shallow HoeffdingTreeClassifier works well and mirrors
        the original implementation's choice.
    window_size
        Number of samples in each of the two rolling windows. Half of it
        is used per window; the windows rotate on drift.
    auc_threshold
        The AUC level under which the windows are considered distinguishable
        and drift is signaled.

    Examples
    --------
    >>> from river import drift, tree

    >>> from river.datasets import synth

    >>> model = drift.D3(
    ...     discriminative_classifier=tree.HoeffdingTreeClassifier(
    ...         grace_period=40, max_depth=3
    ...     )
    ... )

    >>> dataset = synth.Hyperplane(seed=42, n_features=5, n_drift_features=3, mag_change=0.5)

    >>> for i, (x, y) in enumerate(dataset.take(500)):
    ...     model.update(x)
    ...     if model.drift_detected:
    ...         print(f"Change detected at index {i}")
    Change detected at index 399

    """

    _LABEL_FOR_NEW_DATA = True
    _LABEL_FOR_OLD_DATA = False
    _AUC_NUM_THRESHOLDS = 20

    def __init__(
        self,
        discriminative_classifier: base.Classifier,
        window_size: int = 200,
        auc_threshold: float = 0.7,
    ):
        super().__init__()

        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        if not 0.5 < auc_threshold <= 1:
            raise ValueError("auc_threshold must be between 0.5 and 1")

        self.auc_threshold = auc_threshold
        self.window_size = window_size
        self.sub_window_size = window_size // 2
        self.discriminative_classifier = discriminative_classifier

        self._old_window: list[typing.Any] = []
        self._new_window: list[typing.Any] = []
        self._old_window_index = 0
        self._new_window_index = 0
        self._auc = metrics.ROCAUC(n_thresholds=D3._AUC_NUM_THRESHOLDS)  # type: ignore[no-untyped-call]

        self._reset()

    def _reset(self) -> None:
        super()._reset()
        self._old_window = [None] * self.sub_window_size
        self._new_window = [None] * self.sub_window_size
        self._old_window_index = 0
        self._new_window_index = 0
        self._auc = metrics.ROCAUC(n_thresholds=D3._AUC_NUM_THRESHOLDS)  # type: ignore[no-untyped-call]
        self.discriminative_classifier = self.discriminative_classifier.clone()

    def update(self, x: typing.Any) -> None:
        if self._drift_detected:
            self._drift_detected = False

        if self._old_window_index < self.sub_window_size:
            self._old_window[self._old_window_index] = x
            self._old_window_index += 1
            return

        paired_index = self._new_window_index
        paired_sample = self._old_window[paired_index]
        assert paired_sample is not None  # the old window is fully populated before pairing begins
        self._new_window[paired_index] = x

        self.discriminative_classifier.learn_one(x, self._LABEL_FOR_NEW_DATA)
        self.discriminative_classifier.learn_one(paired_sample, self._LABEL_FOR_OLD_DATA)

        prob_new = self.discriminative_classifier.predict_proba_one(x).get(
            self._LABEL_FOR_NEW_DATA, 0.0
        )
        prob_old = self.discriminative_classifier.predict_proba_one(paired_sample).get(
            self._LABEL_FOR_NEW_DATA, 0.0
        )

        self._auc.update(self._LABEL_FOR_NEW_DATA, prob_new)  # type: ignore[no-untyped-call]
        self._auc.update(self._LABEL_FOR_OLD_DATA, prob_old)  # type: ignore[no-untyped-call]

        self._new_window_index += 1

        if self._new_window_index == self.sub_window_size:
            auc_score = self._auc.get()  # type: ignore[no-untyped-call]
            if auc_score > self.auc_threshold or auc_score < self.auc_threshold - 0.5:
                self._drift_detected = True
            self._old_window = list(self._new_window)
            self._new_window_index = 0
            self._auc = metrics.ROCAUC(n_thresholds=D3._AUC_NUM_THRESHOLDS)  # type: ignore[no-untyped-call]
            self.discriminative_classifier = self.discriminative_classifier.clone()
