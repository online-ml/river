import abc


class Recommender(abc.ABC):

    @abc.abstractmethod
    def fit_one(self, row_id, col_id, value):
        pass

    @abc.abstractmethod
    def predict_one(self, row_id, col_id):
        pass
