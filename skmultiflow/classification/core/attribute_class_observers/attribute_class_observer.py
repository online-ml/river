__author__ = 'Jacob Montiel'

from abc import ABCMeta, abstractmethod
from skmultiflow.core.base_object import BaseObject


class AttributeClassObserver(BaseObject, metaclass=ABCMeta):
    """ AttributeClassObserver

    Abstract class for observing the class data distribution for an attribute.
    This observer monitors the class distribution of a given attribute.

    This class should not me instantiated, as none of its methods are implemented.

    Raises
    ------
    NotImplementedError: This is an abstract class.

    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def observe_attribute_class(self, att_val, class_val, weight):
        """ observe_attribute_class

        Updates statistics of this observer given an attribute value, a class
        and the weight of the instance observed

        Parameters
        ----------
        att_val : The value of the attribute

        class_val: The class

        weight: The weight of the instance

        Returns
        -------
        self

        """
        raise NotImplementedError

    @abstractmethod
    def probability_of_attribute_value_given_class(self, att_val, class_val):
        """ probability_of_attribute_value_given_class

        Gets the probability for an attribute value given a class

        Parameters
        ----------
        att_val : The value of the attribute

        class_val: The class

        Returns
        -------
        Probability for an attribute value given a class

        """
        raise NotImplementedError

    @abstractmethod
    def get_best_evaluated_split_suggestion(self, criterion, pre_split_dist, att_idx, binary_only):
        """ get_best_evaluated_split_suggestion

        Gets the best split suggestion given a criterion and a class distribution

        Parameters
        ----------
        criterion: The split criterion to use
        pre_split_dist: The class distribution before the split
        att_idx: The attribute index
        binary_only: True to use binary splits

        Returns
        -------
        Suggestion of best attribute split

        """
        raise NotImplementedError
