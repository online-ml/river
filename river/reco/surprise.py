import surprise

from . import base


class SurpriseWrapper(surprise.AlgoBase):
    """A wrapper to provide compatibility with surprise.

    Parameters
    ----------
    river_recommender

    """

    def __init__(self, river_recommender: base.Recommender):
        self.river_recommender = river_recommender

    def fit(self, trainset):
        surprise.AlgoBase.fit(self, trainset)
        for user, item, y in trainset.all_ratings():
            self.river_recommender.learn_one({"user": user, "item": item}, y)
        return self

    def estimate(self, user, item):
        return self.river_recommender.predict_one({"user": user, "item": item})
