# from river.changepoints.methods.base import ChangePointDetector
from methods.base import ChangePointDetector


class NaiveCPD(ChangePointDetector):
    """
    A change point detector that uses a linear regression over a specified lookback window to predict
    the next value and then compares the prediction with the actual value. If the actual value is too
    far from the prediction, a change point is detected.

    Args:
        lookback_window: Size of the lookback window for linear regression.
        alpha: Threshold parameter for determining change points.

    Methods:
        update(x, t): Update the change point detector with a new data point.
        _reset(): Reset the change point detector to its initial state.
        is_multivariate(): Check if the change point detector is designed for multivariate data.

    """

    def __init__(self, lookback_window, alpha=0.1, **kwargs):
        """
        Initialize the NaiveCPD.

        Args:
            lookback_window: Size of the lookback window for linear regression.
            alpha: Threshold parameter for determining change points.
            **kwargs: Additional keyword arguments.

        """
        super().__init__(**kwargs)
        self.lookback_window = lookback_window
        self.alpha = alpha
        self.lookback_values = []

    def update(self, x, t) -> "ChangePointDetector":
        """
        Update the change point detector with a new data point.

        Args:
            x: The new data point.
            t: The time step or index of the new data point.

        Returns:
            ChangePointDetector: The updated change point detector.

        """
        if t > self.lookback_window:
            # Linear regression without using numpy
            mean_x = sum(range(1, len(self.lookback_values) + 1)) / len(self.lookback_values)
            mean_y = sum(self.lookback_values) / len(self.lookback_values)
            numerator = sum([(i + 1 - mean_x) * (y - mean_y) for i, y in enumerate(self.lookback_values)])
            denominator = sum([(i + 1 - mean_x) ** 2 for i in range(len(self.lookback_values))])
            slope = numerator / denominator
            intercept = mean_y - slope * mean_x
            y_pred = slope * t + intercept

            var_y = sum([(y - mean_y) ** 2 for y in self.lookback_values]) / (len(self.lookback_values) - 1)

            if y_pred - self.alpha * var_y < x < y_pred + self.alpha * var_y:
                self._change_point_detected = False
            else:
                self._change_point_detected = True
            self._change_point_score = abs(x - y_pred) / var_y if var_y > 0 else 0
            self.lookback_values.pop(0)

        self.lookback_values.append(x)
        return self

    def _reset(self):
        """
        Reset the change point detector to its initial state.
        """
        super()._reset()
        self.lookback_values = []

    def is_multivariate(self):
        """
        Check if the change point detector is designed for multivariate data.

        Returns:
            bool: True if the change point detector is designed for multivariate data, False otherwise.
        """
        return False
