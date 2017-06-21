__author__ = 'Guilherme Matsumoto'

from abc import ABCMeta, abstractmethod
from skmultiflow.core.BaseObject import BaseObject

class BaseInstanceStream(BaseObject, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def estimated_remaining_instances(self):
        pass

    @abstractmethod
    def has_more_instances(self):
        pass

    @abstractmethod
    def next_instance(self, batch_size = 1):
        pass

    @abstractmethod
    def is_restartable(self):
        pass

    @abstractmethod
    def restart(self):
        pass

    @abstractmethod
    def has_more_mini_batch(self):
        pass

    @abstractmethod
    def get_num_nominal_attributes(self):
        pass

    @abstractmethod
    def get_num_numerical_attributes(self):
        pass

    @abstractmethod
    def get_num_values_per_nominal_attribute(self):
        pass

    @abstractmethod
    def get_num_attributes(self):
        pass

    @abstractmethod
    def get_num_classes(self):
        pass

    @abstractmethod
    def get_attributes_header(self):
        pass

    @abstractmethod
    def get_classes_header(self):
        pass

    @abstractmethod
    def get_last_instance(self):
        pass

    @abstractmethod
    def prepare_for_use(self):
        pass

    @abstractmethod
    def get_plot_name(self):
        pass

    @abstractmethod
    def get_classes(self):
        pass

    def get_class_type(self):
        return 'stream'

    @abstractmethod
    def get_info(self):
        pass
