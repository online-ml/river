import abc
import numbers

from . import base


class DriftDetector(base.Base):
    """A drift detector."""

    def __init__(self):
        self.drift_detected = False

    def _reset(self):
        """Reset the change detector."""
        self.drift_detected = False

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


class WarningAndDriftDetector(DriftDetector):
    """A drift detector that is also capable of issuing warnings."""

    def __init__(self):
        super().__init__()
        self.warning_detected = False

    def _reset(self):
        """Reset the change detector."""
        super()._reset()
        self.warning_detected = False
