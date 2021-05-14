import copy

from river import base, linear_model, stats, tree

from ..tree.splitter.nominal_splitter_reg import NominalSplitterReg
from .base import Rule


class MeanPredWrapper(base.Regressor):
    def __init__(self):
        self.mean = stats.Mean()

    def learn_one(self, x: dict, y: base.typing.RegTarget, w: int = 1):
        self.mean.update(y, w)

    def predict_one(self, x: dict):
        return self.mean.get()


class RegRule(Rule, base.Regressor):
    def new_nominal_splitter(self):
        return NominalSplitterReg()

    def __init__(self, template_splitter, pred_model):
        super().__init__(template_splitter=template_splitter)
        self._pred_model = pred_model

        self.stats = stats.Var()

    def _update_stats(self, y, w):
        self.stats.update(y, w)

    def learn_one(self, x: dict, y: base.typing.RegTarget, w: int = 1):
        self.update(x, y, w)
        self._pred_model.learn_one(x, y, w)

    def predict_one(self, x: dict):
        return self._pred_model.predict_one(x)


class AMRules(base.Regressor):
    _PRED_MEAN = "mean"
    _PRED_MODEL = "model"
    _VALID_PRED = [_PRED_MEAN, _PRED_MODEL]

    def __init__(
        self,
        grace_period: int = 200,
        pred_type: str = "model",
        pred_model: base.Regressor = None,
        splitter=None,
    ):
        self.grace_period = grace_period

        if splitter is None:
            self.splitter = tree.splitter.EBSTSplitter()
        else:
            self.splitter = splitter

        if pred_type not in self._VALID_PRED:
            raise ValueError(f"Invalid 'pred_type': {pred_type}")
        self.pred_type = pred_type

        self.pred_model = pred_model if pred_model else linear_model.LinearRegression()

        self._default_rule = self._new_rule()

    def _new_rule(self):
        predictor = None
        if self.pred_type == self._PRED_MEAN:
            predictor = MeanPredWrapper()
        else:
            predictor = copy.deepcopy(self.pred_model)

        return RegRule(self.splitter, predictor)
