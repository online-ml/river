import abc
import typing

from river import base


class AnomalyDetector(base.Classifier):
    """An anomaly detector."""

    @property
    def _supervised(self):
        return False

    @property
    def _multiclass(self):
        return False

    @abc.abstractmethod
    def learn_one(
        self, x: dict, y: base.typing.ClfTarget = None, **kwargs
    ) -> "AnomalyDetector":
        """Update the model.

        Parameters
        ----------
        x
            A dictionary of features.

        Returns
        -------
        self

        """

    @abc.abstractmethod
    def score_one(self, x: dict) -> float:
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

    def predict_proba_one(self, x: dict) -> typing.Dict[base.typing.ClfTarget, float]:
        p = self.score_one(x=x)
        return {True: p, False: 1.0 - p}
