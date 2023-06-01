import abc

from river import base


class ChangePointDetector(base.Base):
    """
    An abstract class for change point detection methods. Use this class to be able to run test using the TCPD benchmark.
    """

    def __init__(self, **kwargs):
        self._change_point_detected = False
        self._change_point_score = 0.0

    def _reset(self):
        """
        Reset the change detector.
        """
        self._change_point_detected = False
        self._change_point_score = 0.0

    @property
    def change_point_detected(self) -> bool:
        """
        Returns True if a change point was detected.
        """
        return self._change_point_detected

    @property
    def change_point_score(self) -> float:
        """
        Returns the change point score.
        """
        return self._change_point_score

    @abc.abstractmethod
    def update(self, x, t) -> "ChangePointDetector":
        """Update the change point detector with a single data point.

        Parameters
        ----------
        x
            Input data point.
        t
            Time step. STARTING FROM 1.

        Returns
        -------
        self
        
        """

    @abc.abstractmethod
    def is_multivariate(self):
        """
        Returns True if the change point detector can handle multivariate input sequences.
        """
