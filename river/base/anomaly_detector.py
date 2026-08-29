from __future__ import annotations

import abc
from typing import Any

from river import base

from .estimator import Estimator
from .wrapper import Wrapper

__all__ = ["AnomalyDetector", "SupervisedAnomalyDetector", "AnomalyFilter"]


class AnomalyDetector(Estimator):
    """An anomaly detector."""

    @property
    def _supervised(self) -> bool:
        return False

    @abc.abstractmethod
    def learn_one(self, x: dict[base.typing.FeatureName, Any]) -> None:
        """Update the model.

        Parameters
        ----------
        x
            A dictionary of features.

        """

    @abc.abstractmethod
    def score_one(self, x: dict[base.typing.FeatureName, Any]) -> float:
        """Return an outlier score.

        A high score is indicative of an anomaly. A low score corresponds to a normal observation.

        Parameters
        ----------
        x
            A dictionary of features.

        Returns
        -------
        An anomaly score. A high score is indicative of an anomaly. A low score corresponds a
        normal observation.

        """


class SupervisedAnomalyDetector(Estimator):
    """A supervised anomaly detector."""

    @abc.abstractmethod
    def learn_one(self, x: dict[base.typing.FeatureName, Any], y: base.typing.Target) -> None:
        """Update the model.

        Parameters
        ----------
        x
            A dictionary of features.

        """

    @abc.abstractmethod
    def score_one(self, x: dict[base.typing.FeatureName, Any], y: base.typing.Target) -> float:
        """Return an outlier score.

        A high score is indicative of an anomaly. A low score corresponds a normal observation.

        Parameters
        ----------
        x
            A dictionary of features.

        Returns
        -------
        An anomaly score. A high score is indicative of an anomaly. A low score corresponds a
        normal observation.

        """


class AnomalyFilter(Wrapper[AnomalyDetector], Estimator):
    """Anomaly filter base class.

    An anomaly filter has the ability to classify an anomaly score as anomalous or not. It can then
    be used to filter anomalies, in particular as part of a pipeline.

    Parameters
    ----------
    anomaly_detector
        An anomaly detector wrapped by the anomaly filter.
    protect_anomaly_detector
        Indicates whether or not the anomaly detector should be updated when the anomaly score is
        anomalous. If the data contains sporadic anomalies, then the anomaly detector should likely
        not be updated. Indeed, if it learns the anomaly score, then it will slowly start to
        consider anomalous anomaly scores as normal. This might be desirable, for instance in the
        case of drift.

    """

    def __init__(
        self, anomaly_detector: AnomalyDetector, protect_anomaly_detector: bool = True
    ) -> None:
        self.anomaly_detector = anomaly_detector
        self.protect_anomaly_detector = protect_anomaly_detector

    @property
    def _wrapped_model(self) -> AnomalyDetector:
        return self.anomaly_detector

    @abc.abstractmethod
    def classify(self, score: float) -> bool:
        """Classify an anomaly score as anomalous or not.

        Parameters
        ----------
        score
            An anomaly score to classify.

        Returns
        -------
        A boolean value indicating whether the anomaly score is anomalous or not.

        """

    def score_one(self, *args: Any, **kwargs: Any) -> float:
        """Return an outlier score.

        A high score is indicative of an anomaly. A low score corresponds to a normal observation.

        Parameters
        ----------
        args
            Depends on whether the underlying anomaly detector is supervised or not.

        Returns
        -------
        An anomaly score. A high score is indicative of an anomaly. A low score corresponds a
        normal observation.

        """
        return self.anomaly_detector.score_one(*args, **kwargs)

    def learn_one(self, *args: Any, **learn_kwargs: Any) -> None:
        """Update the anomaly filter and the underlying anomaly detector.

        Parameters
        ----------
        args
            Depends on whether the underlying anomaly detector is supervised or not.

        """
        if self.protect_anomaly_detector and not self.classify(self.score_one(*args)):
            self.anomaly_detector.learn_one(*args, **learn_kwargs)
