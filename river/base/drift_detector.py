import abc
import numbers

from . import base


class DriftDetector(base.Base):
    """A drift detector."""

    def __init__(self):
        self._drift_detected = False

    def _reset(self):
        """Reset the change detector."""
        self._drift_detected = False

    @property
    def drift_detected(self) -> bool:
        """Concept drift alarm.

        True if concept drift is detected.

        """
        return self._drift_detected

    @abc.abstractmethod
    def update(self, x: numbers.Number) -> "DriftDetector":
        """Update the change detector with a single data point.

        Parameters
        ----------
        x
            Input value.

        Returns
        -------
        self

        """
