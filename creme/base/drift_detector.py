from abc import ABCMeta, abstractmethod
from . import estimator


class DriftDetector(estimator.Estimator, metaclass=ABCMeta):
    """ Abstract Drift Detector class.
    """

    @property
    def _supervised(self):
        return False

    def __init__(self):
        super().__init__()
        self._in_concept_change = False
        self._in_warning_zone = False

    def reset(self):
        """Resets the change detector.
        """
        self._in_concept_change = False
        self._in_warning_zone = False

    def detected_change(self):
        """This function returns whether concept drift was detected or not.

        Returns
        -------
        bool
            Whether concept drift was detected or not.

        """
        return self._in_concept_change

    def detected_warning_zone(self):
        """If the change detector supports the warning zone, this function will return
        whether it's inside the warning zone or not.

        Returns
        -------
        bool
            Whether the change detector is in the warning zone or not.

        """
        return self._in_warning_zone

    @abstractmethod
    def add_element(self, input_value):
        """Adds the relevant data from a sample into the change detector.

        Parameters
        ----------
        input_value: Not defined
            Whatever input value the change detector takes.

        Returns
        -------
        bool
            If True, indicates that a drift has been detected

        """
        raise NotImplementedError

    def update(self, value):
        """Update the change detector with a single data point.

        Parameters
        ----------
        value: int, float
            Input value

        Returns
        -------
        tuple
            A tuple (drift, warning) where its elements indicate if a drift or a warning is
            detected.

        """
        return self.add_element(input_value=value)

    def __iadd__(self, other):
        self.update(other)

    @property
    def change_detected(self):
        return self._in_concept_change

    @property
    def in_warning_zone(self):
        return self._in_warning_zone
