import inspect

from river.stats import Var

from ..splitter import EBSTSplitter
from ..splitter.nominal_splitter_reg import NominalSplitterReg
from .leaf import HTLeaf


class LeafMean(HTLeaf):
    """Learning Node for regression tasks that always use the average target
        value as response.

    Parameters
    ----------
    stats
        In regression tasks the node keeps an instance of `river.stats.Var` to estimate
        the target's statistics.
    depth
        The depth of the node.
    splitter
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    kwargs
        Other parameters passed to the learning node.
    """

    def __init__(self, stats, depth, splitter, **kwargs):
        if stats is None:
            # Enforce the usage of Var to keep track of target statistics
            stats = Var()
        super().__init__(stats, depth, splitter, **kwargs)

    @staticmethod
    def new_nominal_splitter():
        return NominalSplitterReg()

    def manage_memory(self, criterion, last_check_ratio, last_check_vr, last_check_e):
        """Trigger Attribute Observers' memory management routines.

        Currently, only `EBSTSplitter` and `TEBSTSplitter` have support to this feature.

        Parameters
        ----------
        criterion
            Split criterion
        last_check_ratio
            The ratio between the second best candidate's merit and the merit of the best
            split candidate.
        last_check_vr
            The best candidate's split merit.
        last_check_e
            Hoeffding bound value calculated in the last split attempt.
        """
        for splitter in self.splitters.values():
            if isinstance(splitter, EBSTSplitter):
                splitter.remove_bad_splits(
                    criterion=criterion,
                    last_check_ratio=last_check_ratio,
                    last_check_vr=last_check_vr,
                    last_check_e=last_check_e,
                    pre_split_dist=self.stats,
                )

    def update_stats(self, y, sample_weight):
        self.stats.update(y, sample_weight)

    def prediction(self, x, *, tree=None):
        return self.stats.mean.get()

    @property
    def total_weight(self):
        """Calculate the total weight seen by the node.

        Returns
        -------
        float
            Total weight seen.

        """
        return self.stats.mean.n

    def calculate_promise(self) -> int:
        """Estimate how likely a leaf node is going to be split.

        Uses the node's depth as a heuristic to estimate how likely the leaf is going to become
        a decision node. The deeper the node is in the tree, the more unlikely it is going to be
        split. To cope with the general tree memory management framework, takes the negative of
        the node's depth as return value. In this way, when sorting the tree leaves by their
        "promise value", the deepest nodes are going to be placed at the first positions as
        candidates to be deactivated.


        Returns
        -------
        int
            The smaller the value, the more unlikely the node is going to be split.

        """
        return -self.depth

    def __repr__(self):
        return f"{repr(self.stats.mean)} | {repr(self.stats)}" if self.stats else ""


class LeafModel(LeafMean):
    """Learning Node for regression tasks that always use a learning model to provide
        responses.

    Parameters
    ----------
    stats
        In regression tasks the node keeps an instance of `river.stats.Var` to estimate
        the target's statistics.
    depth
        The depth of the node.
    splitter
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    leaf_model
        A `river.base.Regressor` instance used to learn from instances and provide
        responses.
    kwargs
        Other parameters passed to the learning node.
    """

    def __init__(self, stats, depth, splitter, leaf_model, **kwargs):
        super().__init__(stats, depth, splitter, **kwargs)

        self._leaf_model = leaf_model
        sign = inspect.signature(leaf_model.learn_one).parameters
        self._model_supports_weights = "sample_weight" in sign or "w" in sign

    def learn_one(self, x, y, *, sample_weight=1.0, tree=None):
        super().learn_one(x, y, sample_weight=sample_weight, tree=tree)

        if self._model_supports_weights:
            self._leaf_model.learn_one(x, y, sample_weight)
        else:
            for _ in range(int(sample_weight)):
                self._leaf_model.learn_one(x, y)

    def prediction(self, x, *, tree=None):
        return self._leaf_model.predict_one(x)


class LeafAdaptive(LeafModel):
    """Learning Node for regression tasks that dynamically selects between predictors and
        might behave as a regression tree node or a model tree node, depending on which predictor
        is the best one.

    Parameters
    ----------
    stats
        In regression tasks the node keeps an instance of `river.stats.Var` to estimate
        the target's statistics.
    depth
        The depth of the node.
    splitter
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    leaf_model
        A `river.base.Regressor` instance used to learn from instances and provide
        responses.
    kwargs
        Other parameters passed to the learning node.
    """

    def __init__(self, stats, depth, splitter, leaf_model, **kwargs):
        super().__init__(stats, depth, splitter, leaf_model, **kwargs)
        self._fmse_mean = 0.0
        self._fmse_model = 0.0

    def learn_one(self, x, y, *, sample_weight=1.0, tree=None):
        pred_mean = self.stats.mean.get()
        pred_model = self._leaf_model.predict_one(x)

        self._fmse_mean = (
            tree.model_selector_decay * self._fmse_mean + (y - pred_mean) ** 2
        )
        self._fmse_model = (
            tree.model_selector_decay * self._fmse_model + (y - pred_model) ** 2
        )

        super().learn_one(x, y, sample_weight=sample_weight, tree=tree)

    def prediction(self, x, *, tree=None):
        if self._fmse_mean < self._fmse_model:  # Act as a regression tree
            return self.stats.mean.get()
        else:  # Act as a model tree
            return super().prediction(x)
