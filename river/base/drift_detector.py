"""Base classes for drift detection.

The _drift_detected and _warning_detected properties are stored as private attributes
and are exposed through the corresponding properties. This is done for documentation
purposes. The properties are not meant to be modified by the user.

"""

from __future__ import annotations

import abc

from . import base


class _BaseDriftDetector(base.Base):
    """Base drift detector.

    This base class is not exposed.

    """

    def __init__(self) -> None:
        self._drift_detected = False

    def _reset(self) -> None:
        """Reset the detector's state."""
        self._drift_detected = False

    @property
    def drift_detected(self) -> bool:
        """Whether or not a drift is detected following the last update."""
        return self._drift_detected


class _BaseDriftAndWarningDetector(_BaseDriftDetector):
    """Base drift detector.

    This base class is not exposed.

    """

    def __init__(self) -> None:
        super().__init__()
        self._warning_detected = False

    def _reset(self) -> None:
        super()._reset()
        self._warning_detected = False

    @property
    def warning_detected(self):
        """Whether or not a drift is detected following the last update."""
        return self._warning_detected


class DriftDetector(_BaseDriftDetector):
    """A drift detector."""

    @abc.abstractmethod
    def update(self, x: int | float) -> None:
        """Update the detector with a single data point.

        Parameters
        ----------
        x
            Input value.

        """


class DriftAndWarningDetector(DriftDetector, _BaseDriftAndWarningDetector):
    """A drift detector that is also capable of issuing warnings."""


class BinaryDriftDetector(_BaseDriftDetector):
    """A drift detector for binary data."""

    @abc.abstractmethod
    def update(self, x: bool) -> None:
        """Update the detector with a single boolean input.

        Parameters
        ----------
        x
            Input boolean.

        """


class BinaryDriftAndWarningDetector(BinaryDriftDetector, _BaseDriftAndWarningDetector):
    """A binary drift detector that is also capable of issuing warnings."""
