import abc

try:
    import surprise
    SURPRISE_INSTALLED = True
except ImportError:
    SURPRISE_INSTALLED = False


if SURPRISE_INSTALLED:

    class Recommender(abc.ABC, surprise.AlgoBase):

        @abc.abstractmethod
        def fit_one(self, row_id, col_id, value):
            pass

        @abc.abstractmethod
        def predict_one(self, row_id, col_id):
            pass

        def fit(self, trainset):
            """Fits to an entire training set.

            This method is compatible with the API from the `surprise` library.

            """
            surprise.AlgoBase.fit(self, trainset)
            for triplet in trainset.all_ratings():
                self.fit_one(*triplet)
            return self

        def estimate(self, *_):
            return self.predict_one(*_)

else:

    class Recommender(abc.ABC, surprise.AlgoBase):

        @abc.abstractmethod
        def fit_one(self, row_id, col_id, value):
            pass

        @abc.abstractmethod
        def predict_one(self, row_id, col_id):
            pass
