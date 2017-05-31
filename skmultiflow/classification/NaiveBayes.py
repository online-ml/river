__author__ = 'Guilherme Matsumoto'

from skmultiflow.classification.base import BaseClassifier
from sklearn.naive_bayes import MultinomialNB

class NaiveBayes(BaseClassifier):

    def __init__(self):
        super().__init__()
        self.classifier = MultinomialNB()
        pass

    def first_fit(self, X, y, classes = None):
        pass

    def fit(self, X, y, classes = None):
        self.classifier.fit(X, y, 1)
        pass

    def partial_fit(self, X, y, classes = None):
        self.classifier.partial_fit(X, y, 1)
        pass

    def predict(self, X):
        return self.classifier.predict(X)


    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def score(self, X, y):
        return self.classifier.score(X, y)