__author__ = 'Guilherme Matsumoto'

from skmultiflow.classification.base import BaseClassifier
from sklearn.linear_model.perceptron import Perceptron

class PerceptronMask(BaseClassifier):

    def __init__(self):
        super().__init__()
        self.classifier = Perceptron(n_iter=50)
        pass

    def fit(self, X, y, classes = None):
        self.classifier.fit(X, y, classes)
        return self

    def partial_fit(self, X, y, classes = None):
        self.classifier.partial_fit(X, y, classes)
        return self

    def predict(self, X):
        return self.classifier.predict(X)


    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def score(self, X, y):
        return self.classifier.score(X, y)

    def get_info(self):
        params = self.classifier.get_params()
        penalty = params['penalty']
        penalty = 'None' if penalty is None else penalty
        fit_int = params['fit_intercept']
        fit_int = 'True' if fit_int else 'False'
        shuffle = params['shuffle']
        shuffle = 'True' if shuffle else 'False'
        return 'Perceptron: penalty: ' + penalty + '  -  alpha: ' + str(round(params['alpha'], 3)) + \
               '  -  fit_intercept: ' + fit_int + '  -  n_iter: ' + str(params['n_iter']) + \
               '  -  shuffle: ' + shuffle
