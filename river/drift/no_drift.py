from __future__ import annotations

from river import base
from river.base.drift_detector import DriftDetector


class NoDrift(base.DriftDetector):
    """Dummy class used to turn off concept drift detection capabilities of adaptive models.


    It always signals that no concept drift was detected.


    Examples
    --------
    >>> from river import drift
    >>> from river import evaluate
    >>> from river import forest
    >>> from river import metrics
    >>> from river.datasets import synth

    >>> dataset = synth.ConceptDriftStream(
    ...     seed=8,
    ...     position=500,
    ...     width=40,
    ... )

    We can turn off the warning detection capabilities of Adaptive Random Forest (ARF) or
    other similar models. Thus, the base models will reset immediately after identifying a drift,
    bypassing the background model building phase:

    >>> model = forest.ARFClassifier(
    ...     leaf_prediction="mc",
    ...     warning_detector=drift.NoDrift(),
    ...     seed=8
    ... )
    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset.take(700), model, metric)
    Accuracy: 76.25%

    >>> model.n_drifts_detected()
    2

    >>> model.n_warnings_detected()
    0

    We can also turn off the concept drift handling capabilities completely:

    >>> stationary_model = forest.ARFClassifier(
    ...     leaf_prediction="mc",
    ...     warning_detector=drift.NoDrift(),
    ...     drift_detector=drift.NoDrift(),
    ...     seed=8
    ... )
    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset.take(700), stationary_model, metric)
    Accuracy: 76.25%

    >>> stationary_model.n_drifts_detected()
    0

    >>> stationary_model.n_warnings_detected()
    0

    """

    def __init__(self):
        super().__init__()

    def update(self, x: int | float) -> DriftDetector:
        return self

    @property
    def drift_detected(self):
        return False
