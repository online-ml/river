import collections

from .. import base


__all__ = ['WeightedMajorityClassifier']


class WeightedMajorityClassifier(collections.UserList, base.Classifier):

    def __init__(self, classifiers, learning_rate=.05):
        super().__init__(classifiers)
        self.learning_rate = learning_rate
        self.weights = [1 for _ in classifiers]

    def fit_one(self, x, y):
        """

        Example:

            ::

                >>> from creme import datasets
                >>> from creme import ensemble
                >>> from creme import linear_model
                >>> from creme import metrics
                >>> from creme import model_selection
                >>> from creme import optim
                >>> from creme import preprocessing

                >>> optimizers = [
                ...     optim.SGD(0.01),
                ...     optim.Adam(),
                ...     optim.AdaGrad()
                ... ]

                >>> for optimizer in optimizers:
                ...
                ...     X_y = datasets.fetch_electricity()
                ...     metric = metrics.Accuracy()
                ...     model = (
                ...         preprocessing.StandardScaler() |
                ...         linear_model.LogisticRegression(optimizer=optimizer)
                ...     )
                ...
                ...     print(optimizer, model_selection.online_score(X_y, model, metric))
                SGD Accuracy: 0.83682
                Adam Accuracy: 0.842933
                AdaGrad Accuracy: 0.809499

                >>> X_y = datasets.fetch_electricity()
                >>> metric = metrics.Accuracy()
                >>> hedge = (
                ...     preprocessing.StandardScaler() |
                ...     ensemble.WeightedMajorityClassifier(
                ...         classifiers=[
                ...             linear_model.LogisticRegression(optimizer=o)
                ...             for o in optimizers
                ...         ]
                ...     )
                ... )

                >>> model_selection.online_score(X_y, hedge, metric)
                Accuracy: 0.843198

        """

        total = 0
        for i, c in enumerate(self):
            # Reduce the weight if the predicted label is not correct
            if c.predict_one(x) != y:
                self.weights[i] *= (1. - self.learning_rate)
            total += self.weights[i]
            c.fit_one(x, y)

        for i, w in enumerate(self.weights):
            self.weights[i] /= total

        return self

    def predict_one(self, x):
        votes = collections.defaultdict(int)
        for i, c in enumerate(self):
            votes[c.predict_one(x)] += self.weights[i]
        return max(votes, key=votes.get)
