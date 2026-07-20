from __future__ import annotations

from river import base


class NotEnoughSamples(base.RiverError):
    """Raised when a statistic hasn't seen enough samples to produce a value."""
