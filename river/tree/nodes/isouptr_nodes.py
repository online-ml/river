from __future__ import annotations

import collections
import functools
import inspect
from copy import deepcopy

from river.stats import Var
from river.utils import VectorDict

from .htr_nodes import LeafMean


class LeafMeanMultiTarget(LeafMean):
    """Learning Node for Multi-target Regression tasks that always uses the mean value
    of the targets as responses.

    Parameters
    ----------
    stats
        In regression tasks the node keeps a `utils.VectorDict` with instances of
        `stats.Var` to estimate the targets' statistics.
    depth
        The depth of the node.
    splitter
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    kwargs
        Other parameters passed to the learning node.
    """

    def __init__(self, stats, depth, splitter, **kwargs):
        stats = stats if stats else VectorDict(default_factory=functools.partial(Var))
        super().__init__(stats, depth, splitter, **kwargs)

    def update_stats(self, y, sample_weight):
        for t in y:
            self.stats[t].update(y[t], sample_weight)

    def prediction(self, x, *, tree=None):
        return {t: self.stats[t].mean.get() if t in self.stats else 0.0 for t in tree.targets}

    @property
    def total_weight(self):
        return list(self.stats.values())[0].mean.n if self.stats else 0

    def __repr__(self):
        if self.stats:
            buffer = "Targets' statistics:"
            for t, var in self.stats.items():
                buffer += f"\n\t{t}: {repr(var.mean)} | {repr(var)}"
            return buffer
        return ""


class LeafModelMultiTarget(LeafMeanMultiTarget):
    """Learning Node for Multi-target Regression tasks that always uses learning models
    for each target.

    Parameters
    ----------
    stats
        In regression tasks the node keeps a `utils.VectorDict` with instances of
        `stats.Var` to estimate the targets' statistics.
    depth
        The depth of the node.
    splitter
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    leaf_models
        A dictionary composed of target identifiers and their respective predictive models.
    kwargs
        Other parameters passed to the learning node.
    """

    def __init__(self, stats, depth, splitter, leaf_models, **kwargs):
        super().__init__(stats, depth, splitter, **kwargs)
        self._leaf_models = leaf_models
        self._model_supports_weights = {}
        if self._leaf_models:
            for t in self._leaf_models:
                sign = inspect.signature(self._leaf_models[t].learn_one).parameters
                self._model_supports_weights[t] = "sample_weight" in sign or "w" in sign

    def learn_one(self, x, y, *, sample_weight=1.0, tree=None):
        super().learn_one(x, y, sample_weight=sample_weight, tree=tree)

        for target_id, y_ in y.items():
            try:
                model = self._leaf_models[target_id]
            except KeyError:
                if isinstance(tree.leaf_model, dict):
                    if target_id in tree.leaf_model:
                        self._leaf_models[target_id] = deepcopy(tree.leaf_model[target_id])
                    else:
                        # Pick the first available model in case not all the targets' models
                        # are defined
                        self._leaf_models[target_id] = deepcopy(
                            next(iter(self._leaf_models.values()))
                        )
                    model = self._leaf_models[target_id]
                else:
                    self._leaf_models[target_id] = deepcopy(tree.leaf_model)
                    model = self._leaf_models[target_id]
                sign = inspect.signature(model.learn_one).parameters
                self._model_supports_weights[target_id] = "sample_weight" in sign or "w" in sign

            # Now the proper training
            if self._model_supports_weights[target_id]:
                model.learn_one(x, y_, sample_weight)
            else:
                for _ in range(int(sample_weight)):
                    model.learn_one(x, y_)

    def prediction(self, x, *, tree=None):
        return {
            t: self._leaf_models[t].predict_one(x) if t in self._leaf_models else 0.0
            for t in tree.targets
        }


class LeafAdaptiveMultiTarget(LeafModelMultiTarget):
    """Learning Node for multi-target regression tasks that dynamically selects between
    predictors and might behave as a regression tree node or a model tree node, depending
    on which predictor is the best one.

    Parameters
    ----------
    stats
        In regression tasks the node keeps a `utils.VectorDict` with instances of
        `stats.Var` to estimate the targets' statistics.
    depth
        The depth of the node.
    splitter
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    leaf_models
        A dictionary composed of target identifiers and their respective predictive models.
    kwargs
        Other parameters passed to the learning node.
    """

    def __init__(self, stats, depth, splitter, leaf_models, **kwargs):
        super().__init__(stats, depth, splitter, leaf_models, **kwargs)
        self._fmse_mean = collections.defaultdict(float)
        self._fmse_model = collections.defaultdict(float)

    def learn_one(self, x, y, *, sample_weight=1.0, tree=None):
        pred_mean = {t: self.stats[t].mean.get() if t in self.stats else 0.0 for t in tree.targets}
        pred_model = super().prediction(x, tree=tree)

        for t in tree.targets:  # Update the faded errors
            self._fmse_mean[t] = (
                tree.model_selector_decay * self._fmse_mean[t] + (y[t] - pred_mean[t]) ** 2
            )
            self._fmse_model[t] = (
                tree.model_selector_decay * self._fmse_model[t] + (y[t] - pred_model[t]) ** 2
            )

        super().learn_one(x, y, sample_weight=sample_weight, tree=tree)

    def prediction(self, x, *, tree=None):
        pred = {}
        for t in tree.targets:
            if self._fmse_mean[t] < self._fmse_model[t]:  # Act as a regression tree
                pred[t] = self.stats[t].mean.get() if t in self.stats else 0.0
            else:  # Act as a model tree
                try:
                    pred[t] = self._leaf_models[t].predict_one(x)
                except KeyError:
                    pred[t] = 0.0
        return pred
