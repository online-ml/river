from abc import ABCMeta, abstractmethod


class InstanceConditionalTest(metaclass=ABCMeta):
    """Abstract class for instance conditional test to split nodes in Hoeffding Trees.

    This class should not me instantiated, as none of its methods are implemented.

    Raises
    ------
    NotImplementedError: This is an abstract class.

    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def branch_for_instance(self, X):
        """Return the number of the branch for an instance, -1 if unknown.

        Parameters
        ----------
        X : Numpy.ndarray of shape (n_samples, n_features)
            The instance to be used.

        Returns
        -------
        int
            The index of the branch for the instance, -1 if unknown.

        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def max_branches():
        """Get the max number branches, -1 if unknown.

        Returns
        -------
        The max number of branches, -1 if unknown.

        """
        raise NotImplementedError

    @abstractmethod
    def describe_condition_for_branch(self, branch):
        """Describe the condition of a branch. It is used to describe the branch.

        Parameters
        ----------
        branch: int
            The index of the branch to describe

        Returns
        -------
        string
            The description of the condition for the branch

        """
        raise NotImplementedError

    @abstractmethod
    def get_atts_test_depends_on(self):
        """Return an array with the attributes that the test depends on.

        Returns
        -------
        array_like
            The attributes that the test depends on.

        """
        raise NotImplementedError
