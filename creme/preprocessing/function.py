from .. import base


__all__ = ['FunctionTransformer']


class FunctionTransformer(base.Transformer):

    def __init__(self, func):
        self.func = func

    def fit_one(self, x, y=None):
        return self.func(x)

    def transform_one(self, x):
        return self.func(x)
