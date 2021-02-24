from abc import ABCMeta, abstractmethod

from river import base

from .._attribute_test import AttributeSplitSuggestion


class Splitter(base.Estimator, metaclass=ABCMeta):
    """Base class for the tree splitters.

    Attribute Observers (AO) or Splitters are responsible by monitoring an input feature
    and evaluating possible split points to them. They can also perform other tasks related
    to the monitored feature, such as estimating the feature's probability density
    function.

    This class should not be instantiated, as none of its methods are implemented.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def update(self, att_val, target_val, sample_weight) -> "Splitter":
        """Update statistics of this observer given an attribute value, its target value
        and the weight of the instance observed.

        Parameters
        ----------
        att_val
            The value of the monitored attribute.
        target_val
            The target value.
        sample_weight
            The weight of the instance.
        """

    @abstractmethod
    def cond_proba(self, att_val, class_val) -> float:
        """Get the probability for an attribute value given a class.

        Parameters
        ----------
        att_val
            The value of the attribute.
        class_val
            The class value.

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

    @property
    def is_numeric(self) -> bool:
        return True

    @property
    def is_target_class(self) -> bool:
        return True
