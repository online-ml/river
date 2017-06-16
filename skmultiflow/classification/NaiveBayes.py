__author__ = 'Guilherme Matsumoto'

from skmultiflow.classification.base import BaseClassifier
from sklearn.naive_bayes import MultinomialNB

class NaiveBayes(BaseClassifier):

    def __init__(self):
        super().__init__()
        self.classifier = MultinomialNB()
        pass

    def fit(self, X, y, classes = None):
        self.classifier.fit(X, y, classes)
        return self

    def partial_fit(self, X, y, classes = None, warmstart = False):
        if warmstart:
            self.classifier.partial_fit(X, y, classes)
        else:
            self.classifier.partial_fit(X, y, classes)
        return self

    def predict(self, X):
        return self.classifier.predict(X)


    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def score(self, X, y):
        return self.classifier.score(X, y)