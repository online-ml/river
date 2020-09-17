from abc import ABCMeta, abstractmethod


class AttributeObserver(metaclass=ABCMeta):
    """Abstract class for observing the class data distribution for an attribute.
    This observer monitors the class distribution of a given attribute.

    This class should not be instantiated, as none of its methods are implemented.

    Raises
    ------
    NotImplementedError: This is an abstract class.

    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def update(self, att_val, class_val, weight):
        """Update statistics of this observer given an attribute value, a class
        and the weight of the instance observed.

        Parameters
        ----------
        att_val : float
            The value of the attribute.

        class_val: int
            The class value.

        weight: float
            The weight of the instance.

        """
        raise NotImplementedError

    @abstractmethod
    def probability_of_attribute_value_given_class(self, att_val, class_val):
        """Get the probability for an attribute value given a class.

        Parameters
        ----------
        att_val: float
            The value of the attribute.

        class_val: int
            The class value.

        Returns
        -------
        float
            Probability for an attribute value given a class.

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
