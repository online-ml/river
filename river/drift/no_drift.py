from __future__ import annotations

from river import base


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
    ... ).take(700)

    We can turn off the warning detection capabilities of Adaptive Random Forest (ARF) or
    other similar models. Thus, the base models will reset immediately after identifying a drift,
    bypassing the background model building phase:

    >>> adaptive_model = forest.ARFClassifier(
    ...     leaf_prediction="mc",
    ...     warning_detector=drift.NoDrift(),
    ...     seed=8
    ... )

    We can also turn off the concept drift handling capabilities completely:

    >>> stationary_model = forest.ARFClassifier(
    ...     leaf_prediction="mc",
    ...     warning_detector=drift.NoDrift(),
    ...     drift_detector=drift.NoDrift(),
    ...     seed=8
    ... )

    Let's put that to test:

    >>> for x, y in dataset:
    ...     adaptive_model.learn_one(x, y)
    ...     stationary_model.learn_one(x, y)

    The adaptive model:

    >>> adaptive_model.n_drifts_detected()
    2

    >>> adaptive_model.n_warnings_detected()
    0

    The stationary one:

    >>> stationary_model.n_drifts_detected()
    0

    >>> stationary_model.n_warnings_detected()
    0

    """

    def __init__(self):
        super().__init__()

    def update(self, x: int | float): ...

    @property
    def drift_detected(self):
        return False
