"""Base classes for drift detection.

The _drift_detected and _warning_detected properties are stored as private attributes
and are exposed through the corresponding properties. This is done for documentation
purposes. The properties are not meant to be modified by the user.

"""
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
    def drift_detected(self):
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


class DriftAndWarningDetector(DriftDetector):
    """A drift detector that is also capable of issuing warnings."""

    def __init__(self):
        super().__init__()
        self._warning_detected = False

    def _reset(self):
        """Reset the change detector."""
        super()._reset()
        self._warning_detected = False

    @property
    def warning_detected(self):
        return self._warning_detected
