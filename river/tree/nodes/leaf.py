import abc
import copy
import numbers
import typing

from ..base import Leaf
from ..utils import BranchFactory


class HTLeaf(Leaf, metaclass=abc.ABCMeta):
    """Base leaf class to be used in Hoeffding Trees.

    Parameters
    ----------
    stats
        Target statistics (they differ in classification and regression tasks).
    depth
        The depth of the node
    splitter
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    attributes
        Other parameters passed to the learning node.
    """

    def __init__(self, stats, depth, splitter, **kwargs):
        super().__init__(**kwargs)
        self.stats = stats
        self.depth = depth

        self.splitter = splitter

        self.splitters = {}
        self._disabled_attrs = set()
        self._last_split_attempt_at = self.total_weight

    @property
    @abc.abstractmethod
    def total_weight(self) -> float:
        pass

    def is_active(self):
        return self.splitters is not None

    def activate(self):
        if not self.is_active():
            self.splitters = {}

    def deactivate(self):
        self.splitters = None

    @property
    def last_split_attempt_at(self) -> float:
        """The weight seen at last split evaluation.

        Returns
        -------
        Weight seen at last split evaluation.
        """
        return self._last_split_attempt_at

    @last_split_attempt_at.setter
    def last_split_attempt_at(self, weight):
        """Set the weight seen at last split evaluation.

        Parameters
        ----------
        weight
            Weight seen at last split evaluation.
        """
        self._last_split_attempt_at = weight

    @staticmethod
    @abc.abstractmethod
    def new_nominal_splitter():
        pass

    @abc.abstractmethod
    def update_stats(self, y, sample_weight):
        pass

    def _iter_features(self, x) -> typing.Iterable:
        """Determine how the input instance is looped through when updating the splitters.

        Parameters
        ----------
        x
            The input instance.
        """
        for att_id, att_val in x.items():
            yield att_id, att_val

    def update_splitters(self, x, y, sample_weight, nominal_attributes):
        for att_id, att_val in self._iter_features(x):
            if att_id in self._disabled_attrs:
                continue

            try:
                splitter = self.splitters[att_id]
            except KeyError:
                if (
                    nominal_attributes is not None and att_id in nominal_attributes
                ) or not isinstance(att_val, numbers.Number):
                    splitter = self.new_nominal_splitter()
                else:
                    splitter = copy.deepcopy(self.splitter)

                self.splitters[att_id] = splitter
            splitter.update(att_val, y, sample_weight)

    def best_split_suggestions(self, criterion, tree) -> typing.List[BranchFactory]:
        """Find possible split candidates.

        Parameters
        ----------
        criterion
            The splitting criterion to be used.
        tree
            Decision tree.

        Returns
        -------
        Split candidates.
        """
        best_suggestions = []
        pre_split_dist = self.stats
        if tree.merit_preprune:
            # Add null split as an option
            null_split = BranchFactory()
            best_suggestions.append(null_split)
        for att_id, splitter in self.splitters.items():
            best_suggestion = splitter.best_evaluated_split_suggestion(
                criterion, pre_split_dist, att_id, tree.binary_split
            )
            best_suggestions.append(best_suggestion)

        return best_suggestions

    def disable_attribute(self, att_id):
        """Disable an attribute observer.

        Parameters
        ----------
        att_id
            Attribute index.

        """
        if att_id in self.splitters:
            del self.splitters[att_id]
            self._disabled_attrs.add(att_id)

    def learn_one(self, x, y, *, sample_weight=1.0, tree=None):
        """Update the node with the provided sample.

        Parameters
        ----------
        x
            Sample attributes for updating the node.
        y
            Target value.
        sample_weight
            Sample weight.
        tree
            Tree to update.

        Notes
        -----
        This base implementation defines the basic functioning of a learning node.
        All classes overriding this method should include a call to `super().learn_one`
        to guarantee the learning process happens consistently.
        """
        self.update_stats(y, sample_weight)
        if self.is_active():
            self.update_splitters(x, y, sample_weight, tree.nominal_attributes)

    @abc.abstractmethod
    def prediction(self, x, *, tree=None) -> dict:
        pass

    @abc.abstractmethod
    def calculate_promise(self) -> int:
        """Calculate node's promise.

        Returns
        -------
        int
            A small value indicates that the node has seen more samples of a
            given class than the other classes.

        """
