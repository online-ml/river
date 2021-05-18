import typing
from abc import ABCMeta, abstractmethod

from river import base

from ..split_criterion.base_split_criterion import SplitCriterion
from ..utils import BranchFactory


class Splitter(base.Estimator, metaclass=ABCMeta):
    """Base class for the tree splitters.

    Each Attribute Observer (AO) or Splitter monitors one input feature and finds the best
    split point for this attribute. AOs can also perform other tasks related to the monitored
    feature, such as estimating its probability density function (classification case).

    This class should not be instantiated, as none of its methods are implemented.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def update(self, att_val, target_val: base.typing.Target, sample_weight: float):
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
    def cond_proba(self, att_val, target_val: base.typing.ClfTarget) -> float:
        """Get the probability for an attribute value given a class.

        Parameters
        ----------
        att_val
            The value of the attribute.
        target_val
            The target (class label) value.

        Returns
        -------
            Probability for an attribute value given a class.
        """

    @abstractmethod
    def best_evaluated_split_suggestion(
        self,
        criterion: SplitCriterion,
        pre_split_dist: typing.Union[typing.List, typing.Dict],
        att_idx: base.typing.FeatureName,
        binary_only: bool,
    ) -> BranchFactory:
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
        """Determine whether or not the splitter works with numerical features."""
        return True

    @property
    def is_target_class(self) -> bool:
        """Check on which kind of learning task the splitter is designed to work.

        If `True`, the splitter works with classification trees, otherwise it is designed for
        regression trees.
        """
        return True
