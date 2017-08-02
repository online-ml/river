__author__ = 'Guilherme Matsumoto'

from abc import ABCMeta, abstractmethod
from skmultiflow.core.base_object import BaseObject


class BaseDriftDetector(BaseObject, metaclass=ABCMeta):
    """ Abstract Drift Detector
    
    Any drift detector class should follow this minimum structure in 
    order to allow interchangeability between all change detection 
    methods.
    
    Raises
    ------
    NotImplementedError. All child classes should implement the 
    get_info function.
    
    """
    def __init__(self):
        super().__init__()
        self.in_concept_change = None
        self.in_warning_zone = None
        self.estimation = None
        self.delay = None

    @abstractmethod
    def reset(self):
        self.in_concept_change = False
        self.in_warning_zone = False
        self.estimation = 0.0
        self.delay = 0.0

    def detected_change(self):
        return self.in_concept_change

    def detected_warning_zone(self):
        return self.in_warning_zone

    def get_length_estimation(self):
        return self.estimation

    @abstractmethod
    def add_element(self, input_value):
        pass

    @abstractmethod
    def get_info(self):
        raise NotImplementedError

    def get_class_type(self):
        return 'drift_detector'
