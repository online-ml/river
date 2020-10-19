from copy import deepcopy
from collections import defaultdict

from .base import InactiveLeaf
from .htr_nodes import ActiveLeafRegressor
from .htr_nodes import LearningNodeMean


class LearningNodeMeanMultiTarget(LearningNodeMean):
    def __init__(self, initial_stats, depth):
        super().__init__(initial_stats, depth)

    def predict_one(self, X, *, tree=None):
        return (self.stats[1] / self.stats[0]).to_dict if self.stats else {
            t: 0. for t in tree._targets}


class LearningNodeModelMultiTarget(LearningNodeMeanMultiTarget):
    def __init__(self, initial_stats, depth, leaf_models):
        super().__init__(initial_stats, depth)
        self._leaf_models = leaf_models

    def learn_one(self, x, y, *, sample_weight=1.0, tree=None):
        super().learn_one(x, y, sample_weight=sample_weight, tree=tree)

        for target_id, y_ in y.items():
            try:
                model = self._leaf_models[target_id]
            except KeyError:
                if isinstance(tree.leaf_model, dict):
                    if target_id in tree.leaf_model:
                        # TODO: change to appropriate 'clone' method
                        self._leaf_models[target_id] = tree.leaf_model[target_id].__class__(
                            **tree.leaf_model[target_id]._get_params())
                    else:
                        # Pick the first available model in case not all the targets' models
                        # are defined
                        self._leaf_models[target_id] = deepcopy(
                            next(iter(self.leaf_models.values()))
                        )
                    model = self._leaf_models[target_id]
                else:
                    # TODO: change to appropriate 'clone' method
                    self._leaf_models[target_id] = tree.leaf_model.__class__(
                        **tree.leaf_model._get_params())
                    model = self._leaf_models[target_id]

            # Now the proper training
            try:
                model.learn_one(x, y_, sample_weight)
            except TypeError:  # Learning model does not support weights
                for _ in range(int(sample_weight)):
                    model.learn_one(x, y_)

    def predict_one(self, x, *, tree=None):
        return {
            t: self._leaf_models[t].predict_one(x) if t in self._leaf_models else 0.
            for t in tree._targets
        }


class LearningNodeAdaptiveMultiTarget(LearningNodeModelMultiTarget):
    def __init__(self, initial_stats, depth, leaf_models):
        super().__init__(initial_stats, depth, leaf_models)
        self._fmse_mean = defaultdict(lambda: 0.)
        self._fmse_model = defaultdict(lambda: 0.)

    def learn_one(self, x, y, *, sample_weight=1.0, tree=None):
        pred_mean = (self.stats[1] / self.stats[0]).to_dict() if self.stats else {
            t: 0. for t in tree._targets}
        pred_model = super().predict_one(x, tree=tree)

        for t in tree._targets:  # Update the faded errors
            self._fmse_mean[t] = tree.model_selector_decay * self._fmse_mean[t] \
                + (y[t] - pred_mean[t]) ** 2
            self._fmse_model[t] = tree.model_selector_decay * self._fmse_model[t] \
                + (y[t] - pred_model[t]) ** 2

        super().learn_one(x, y, sample_weight=sample_weight, tree=tree)

    def predict_one(self, x, *, tree=None):
        pred = {}
        for t in tree._targets:
            if self._fmse_mean[t] < self._fmse_model[t]:  # Act as a regression tree
                pred[t] = self.stats[1][t] / self.stats[0] if self.stats else 0.
            else:  # Act as a model tree
                try:
                    pred[t] = self._leaf_models[t].predict_one(x)
                except KeyError:
                    pred[t] = 0.
        return pred


class ActiveLearningNodeMeanMultiTarget(LearningNodeMeanMultiTarget, ActiveLeafRegressor):
    """Learning Node for Multi-target Regression tasks that always uses the mean value
    of the targets as responses.

    Parameters
    ----------
    initial_stats
        In regression tasks this dictionary carries the sufficient to perform
        online variance calculation. They refer to the number of observations
        (key '0'), the sum of the target values (key '1'), and the sum of the
        squared target values (key '2').
    depth
        The depth of the node.
    """
    def __init__(self, initial_stats, depth):
        super().__init__(initial_stats, depth)


class InactiveLearningNodeMeanMultiTarget(LearningNodeMeanMultiTarget, InactiveLeaf):
    """Inactive Learning Node for Multi-target Regression tasks that always uses
    the mean value of the targets as responses.

    Parameters
    ----------
    initial_stats
        In regression tasks this dictionary carries the sufficient to perform
        online variance calculation. They refer to the number of observations
        (key '0'), the sum of the target values (key '1'), and the sum of the
        squared target values (key '2').
    depth
        The depth of the node.
    """
    def __init__(self, initial_stats, depth):
        super().__init__(initial_stats, depth)


class ActiveLearningNodeModelMultiTarget(LearningNodeModelMultiTarget, ActiveLeafRegressor):
    """Learning Node for Multi-target Regression tasks that always uses learning models
    for each target.

    Parameters
    ----------
    initial_stats
        In regression tasks this dictionary carries the sufficient statistics
        to perform online variance calculation. They refer to the number of
        observations (key '0'), the sum of the target values (key '1'), and
        the sum of the squared target values (key '2').
    depth
        The depth of the node.
    leaf_models
        A dictionary composed of target identifiers and their respective predictive models.
    """
    def __init__(self, initial_stats, depth, leaf_models):
        super().__init__(initial_stats, depth, leaf_models)


class InactiveLearningNodeModelMultiTarget(LearningNodeModelMultiTarget, InactiveLeaf):
    """Inactive Learning Node for Multi-target Regression tasks that always uses
    learning models for each target.

    Parameters
    ----------
    initial_stats
        In regression tasks this dictionary carries the sufficient statistics
        to perform online variance calculation. They refer to the number of
        observations (key '0'), the sum of the target values (key '1'), and
        the sum of the squared target values (key '2').
    depth
        The depth of the node.
    leaf_models
        A dictionary composed of target identifiers and their respective predictive models.
    """
    def __init__(self, initial_stats, depth, leaf_models):
        super().__init__(initial_stats, depth, leaf_models)


class ActiveLearningNodeAdaptiveMultiTarget(LearningNodeAdaptiveMultiTarget, ActiveLeafRegressor):
    """Learning Node for multi-target regression tasks that dynamically selects between
    predictors and might behave as a regression tree node or a model tree node, depending
    on which predictor is the best one.

    Parameters
    ----------
    initial_stats
        In regression tasks this dictionary carries the sufficient statistics
        to perform online variance calculation. They refer to the number of
        observations (key '0'), the sum of the target values (key '1'), and
        the sum of the squared target values (key '2').
    depth
        The depth of the node.
    leaf_models
        A dictionary composed of target identifiers and their respective predictive models.
    """
    def __init__(self, initial_stats, depth, leaf_models):
        super().__init__(initial_stats, depth, leaf_models)


class InactiveLearningNodeAdaptiveMultiTarget(LearningNodeAdaptiveMultiTarget, InactiveLeaf):
    """Inactive Learning Node for multi-target regression tasks that dynamically selects
    between predictors and might behave as a regression tree node or a model tree node,
    depending on which predictor is the best one.

    Parameters
    ----------
    initial_stats
        In regression tasks this dictionary carries the sufficient statistics
        to perform online variance calculation. They refer to the number of
        observations (key '0'), the sum of the target values (key '1'), and
        the sum of the squared target values (key '2').
    depth
        The depth of the node.
    leaf_models
        A dictionary composed of target identifiers and their respective predictive models.
    """
    def __init__(self, initial_stats, depth, leaf_models):
        super().__init__(initial_stats, depth, leaf_models)
