from . import base


__all__ = ['SVD']


class SVD(base.Recommender):

    def fit_one(self, row_id, col_id, value):
        pass

    def predict_one(self, row_id, col_id):
        pass
