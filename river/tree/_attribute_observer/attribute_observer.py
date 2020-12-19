from abc import ABCMeta, abstractmethod

from .._attribute_test import AttributeSplitSuggestion


class AttributeObserver(metaclass=ABCMeta):
    """Abstract class for observing the class data distribution for an attribute.
    This observer monitors the class distribution of a given attribute.

    This class should not be instantiated, as none of its methods are implemented.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def update(self, att_val, target_val, sample_weight) -> "AttributeObserver":
        """Update statistics of this observer given an attribute value, its target value
        and the weight of the instance observed.

        Parameters
        ----------
        att_val
            The value of the attribute.
        target_val
            The target value.
        sample_weight
            The weight of the instance.
        """

    @abstractmethod
    def probability_of_attribute_value_given_class(self, att_val, target_val) -> float:
        """Get the probability for an attribute value given a class.

        Parameters
        ----------
        att_val
            The value of the attribute.
        target_val
            The target value.

        Returns
        -------
            Probability for an attribute value given a class.
        """

    @abstractmethod
    def best_evaluated_split_suggestion(
        self, criterion, pre_split_dist, att_idx, binary_only
    ) -> AttributeSplitSuggestion:
        """Get the best split suggestion given a criterion and the target's statistics.

        Parameters
        ----------
        criterion
            The split criterion to use.
        pre_split_dist
            The target statistics before the split.
        att_idx
            The attribute index.
        binary_only
            True if only binary splits are allowed.

        Returns
        -------
            Suggestion of the best attribute split.
        """
