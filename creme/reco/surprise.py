import surprise

from . import base


class SurpriseWrapper(surprise.AlgoBase):
    """A wrapper to provide compatibility with surprise.

    Parameters:
        creme_recommender

    """

    def __init__(self, creme_recommender: base.Recommender):
        self.creme_recommender = creme_recommender

    def fit(self, trainset):
        surprise.AlgoBase.fit(self, trainset)
        for user, item, y in trainset.all_ratings():
            self.creme_recommender.fit_one({'user': user, 'item': item}, y)
        return self

    def estimate(self, user, item):
        return self.creme_recommender.predict_one({'user': user, 'item': item})
