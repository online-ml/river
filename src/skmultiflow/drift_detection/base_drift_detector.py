from abc import ABCMeta, abstractmethod
from skmultiflow.core import BaseSKMObject


class BaseDriftDetector(BaseSKMObject, metaclass=ABCMeta):
    """ Abstract Drift Detector

    Any drift detector class should follow this minimum structure in
    order to allow interchangeability between all change detection
    methods.

    Raises
    ------
    NotImplementedError. All child classes should implement the
    get_info function.

    """

    estimator_type = "drift_detector"

    def __init__(self):
        super().__init__()
        self.in_concept_change = None
        self.in_warning_zone = None
        self.estimation = None
        self.delay = None

    def reset(self):
        """ reset

        Resets the change detector parameters.

        """
        self.in_concept_change = False
        self.in_warning_zone = False
        self.estimation = 0.0
        self.delay = 0.0

    def detected_change(self):
        """ detected_change

        This function returns whether concept drift was detected or not.

        Returns
        -------
        bool
            Whether concept drift was detected or not.

        """
        return self.in_concept_change

    def detected_warning_zone(self):
        """ detected_warning_zone

        If the change detector supports the warning zone, this function will return
        whether it's inside the warning zone or not.

        Returns
        -------
        bool
            Whether the change detector is in the warning zone or not.

        """
        return self.in_warning_zone

    def get_length_estimation(self):
        """ get_length_estimation

        Returns the length estimation.

        Returns
        -------
        int
            The length estimation

        """
        return self.estimation

    @abstractmethod
    def add_element(self, input_value):
        """ add_element

        Adds the relevant data from a sample into the change detector.

        Parameters
        ----------
        input_value: Not defined
            Whatever input value the change detector takes.

        Returns
        -------
        BaseDriftDetector
            self, optional

        """
        raise NotImplementedError
