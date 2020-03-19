import surprise


class SurpriseWrapper(surprise.AlgoBase):

    def __init__(self, creme_recommender):
        self.creme_recommender = creme_recommender

    def fit(self, trainset):
        """Fits to an entire training set.

        This method is compatible with the API from the `surprise` library.

        """
        surprise.AlgoBase.fit(self, trainset)
        for user, item, y in trainset.all_ratings():
            self.creme_recommender.fit_one({'user': user, 'item': item}, y)
        return self

    def estimate(self, user, item):
        return self.creme_recommender.predict_one({'user': user, 'item': item})
