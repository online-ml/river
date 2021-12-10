import abc
import typing

from river import base


class AnomalyDetector(base.Estimator):
    """An anomaly detector."""

    @property
    def _supervised(self):
        return False

    @abc.abstractmethod
    def learn_one(self, x: dict) -> "AnomalyDetector":
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
        """Predict the probability of each label for a dictionary of features `x`.

                Parameters
                ----------
                x
                    A dictionary of features.

                Returns
                -------
                A dictionary that associates a probability which each label.

                """

        p = self.score_one(x=x)
        return {True: p, False: 1.0 - p}
