# from river.changepoints.methods import ChangePointDetector TODO: Change path for integration into river
from .base import ChangePointDetector


class ZeroPredictor(ChangePointDetector):
    """A change point detector that never predicts a change point except for the first sample."""

    def __init__(self):
        super().__init__()

    def update(self, x, t) -> "ChangePointDetector":
        # This predictor never predicts a change point
        if t == 1:
            self._change_point_detected = True
        else:
            self._change_point_detected = False
        return self

    def is_multivariate(self):
        return True
