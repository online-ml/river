__author__ = 'Jacob Montiel'

from abc import ABCMeta, abstractmethod
from skmultiflow.core.base_object import BaseObject


class InstanceConditionalTest(metaclass=ABCMeta):
    """ InstanceConditionalTest

    Abstract class for instance conditional test to split nodes in Hoeffding Trees.

    This class should not me instantiated, as none of its methods are implemented.

    Raises
    ------
    NotImplementedError: This is an abstract class.

    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def branch_for_instance(self, X):
        """ branch_for_instance

        Returns the number of the branch for an instance, -1 if unknown.

        Parameters
        ----------
        X : The instance to be used

        Returns
        -------
        The number of the branch for an instance, -1 if unknown.

        """
        raise NotImplementedError

    @abstractmethod
    def max_branches(self):
        """ branch_for_instance

        Gets the number of maximum branches, -1 if unknown.

        Parameters
        ----------
        self

        Returns
        -------
        The number of maximum branches, -1 if unknown.

        """
        raise NotImplementedError

    @abstractmethod
    def describe_condition_for_branch(self, branch):
        """ describe_condition_for_branch

        Gets the text that describes the condition of a branch. It is used to describe the branch.

        Parameters
        ----------
        branch: The number of the branch to describe

        Returns
        -------
        The text that describes the condition of the branch

        """
        raise NotImplementedError

    @abstractmethod
    def get_atts_test_depends_on(self):
        """ get_atts_test_depends_on

        Returns an array with the attributes that the test depends on.

        Parameters
        ----------
        self

        Returns
        -------
        An array with the attributes that the test depends on.

        """
        raise NotImplementedError