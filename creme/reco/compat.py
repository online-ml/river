try:
    import surprise
    SURPRISE_INSTALLED = True
except ImportError:
    SURPRISE_INSTALLED = False


if SURPRISE_INSTALLED:

    class SurpriseWrapper(surprise.AlgoBase):

        def __init__(self, creme_recommender):
            self.creme_recommender = creme_recommender

        def fit(self, trainset):
            """Fits to an entire training set.

            This method is compatible with the API from the `surprise` library.

            """
            surprise.AlgoBase.fit(self, trainset)
            for triplet in trainset.all_ratings():
                self.creme_recommender.fit_one(*triplet)
            return self

        def estimate(self, *_):
            return self.creme_recommender.predict_one(*_)
