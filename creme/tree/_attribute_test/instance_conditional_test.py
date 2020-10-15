from abc import ABCMeta, abstractmethod


class InstanceConditionalTest(metaclass=ABCMeta):
    """Abstract class for instance conditional test to split nodes in Hoeffding Trees.

    This class should not me instantiated, as none of its methods are implemented.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def branch_for_instance(self, x: dict):
        """Return the number of the branch for an instance, -1 if unknown.

        Parameters
        ----------
        x
            The instance to be used.

        Returns
        -------
        int
            The index of the branch for the instance, -1 if unknown.

        """

    @staticmethod
    @abstractmethod
    def max_branches():
        """Get the max number branches, -1 if unknown.

        Returns
        -------
        The max number of branches, -1 if unknown.

        """

    @abstractmethod
    def describe_condition_for_branch(self, branch: int):
        """Describe the condition of a branch. It is used to describe the branch.

        Parameters
        ----------
        branch
            The index of the branch to describe

        Returns
        -------
        string
            The description of the condition for the branch

        """

    @abstractmethod
    def get_atts_test_depends_on(self):
        """Return an array with the attributes that the test depends on.

        Returns
        -------
        array_like
            The attributes that the test depends on.

        """
