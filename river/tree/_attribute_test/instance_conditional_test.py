from abc import ABCMeta, abstractmethod
import typing


class InstanceConditionalTest(metaclass=ABCMeta):
    """Abstract class for instance conditional test to split nodes in Hoeffding Trees.

    This class should not me instantiated, as none of its methods are implemented.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def branch_for_instance(self, x: dict) -> int:
        """Return the number of the branch for an instance, -1 if unknown.

        Parameters
        ----------
        x
            The instance to be used.

        Returns
        -------
        The index of the branch for the instance, -1 if unknown.

        """

    @staticmethod
    @abstractmethod
    def max_branches() -> int:
        """Get the max number branches, -1 if unknown.

        Returns
        -------
        The max number of branches, -1 if unknown.
        """

    @abstractmethod
    def describe_condition_for_branch(self, branch: int, shorten=False) -> str:
        """Describe the condition of a branch. It is used to describe the branch.

        Parameters
        ----------
        branch
            The index of the branch to describe
        shorten
            Whether or not omit the feature name from the description.

        Returns
        -------
        The description of the condition for the branch

        """

    @abstractmethod
    def attrs_test_depends_on(self) -> typing.List:
        """Return an array with the attributes that the test depends on.

        Returns
        -------
        The attributes that the test depends on.
        """
