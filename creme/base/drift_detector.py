import numbers
from abc import ABCMeta, abstractmethod

from . import estimator


class DriftDetector(estimator.Estimator, metaclass=ABCMeta):
    """Abstract Drift Detector class.
    """

    @property
    def _supervised(self):
        return False

    def __init__(self):
        super().__init__()
        self._in_concept_change = False
        self._in_warning_zone = False

    def reset(self):
        """Reset the change detector.
        """
        self._in_concept_change = False
        self._in_warning_zone = False

    @property
    def change_detected(self):
        """Concept Drift alarm.

        Returns
        -------
        bool
            True if concept drift is detected.

        """
        return self._in_concept_change

    @property
    def warning_detected(self):
        """Warning zone alarm.

        Indicates if the drift detector is in the warning zone.
        Applicability depends on each drift detector implementation.

        Returns
        -------
        bool
            True if the change detector is in the warning zone.

        """
        return self._in_warning_zone

    @abstractmethod
    def update(self, value: numbers.Number):
        """Update the change detector with a single data point.

        Parameters
        ----------
        value: Input value

        Returns
        -------
        tuple
            A tuple (drift, warning) where its elements indicate if a drift or a warning is
            detected.

        """
        raise NotImplementedError

    def __iadd__(self, other):
        self.update(other)
