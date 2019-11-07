from abc import ABCMeta, abstractmethod


class AdaNode(metaclass=ABCMeta):
    """
        Abstract Class to create a New Node for the HoeffdingAdaptiveTree (HAT)
    """

    @abstractmethod
    def number_leaves(self):
        pass

    @abstractmethod
    def get_error_estimation(self):
        pass

    @abstractmethod
    def get_error_width(self):
        pass

    @abstractmethod
    def is_null_error(self):
        pass

    @abstractmethod
    def kill_tree_children(self, hat):
        pass

    @abstractmethod
    def learn_from_instance(self, X, y, weight, hat, parent, parent_branch):
        pass

    @abstractmethod
    def filter_instance_to_leaves(self, X, y, weight, parent, parent_branch,
                                  update_splitter_counts, found_nodes=None):
        pass
