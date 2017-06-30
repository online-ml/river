__author__ = 'Jesse Read'

import copy as cp

from numpy import *
from sklearn import linear_model

from skmultiflow.classification.base import BaseClassifier
from skmultiflow.evaluation.metrics.metrics import *


class MultiOutputLearner(BaseClassifier) :
    '''
        Multi-Output Learner
        --------------------
        A Meta Learner. Does either classifier or regression, depending on base learner 
        (which by default is LogisticRegression).
    '''

    h = None
    L = -1

    def __init__(self, h=linear_model.SGDClassifier(n_iter=100)):
        super().__init__()
        self.hop = h
        self.h = None
        self.L = None

    def configure(self):
        self.h = [cp.deepcopy(self.hop) for j in range(self.L)]

    def fit(self, X, Y, classes = None):
        N,L = Y.shape
        self.L = L
        self.h = [cp.deepcopy(self.hop) for j in range(self.L)]

        for j in range(self.L):
            self.h[j].fit(X, Y[:,j])
        return self

    def partial_fit(self, X, Y, classes=None):
        N,self.L = Y.shape

        if self.h is None:
            self.configure()

        for j in range(self.L):
            self.h[j].partial_fit(X, Y[:,j], classes)

        return self

    def predict(self, X):
        '''
            return predictions for X
        '''
        N,D = X.shape
        Y = zeros((N,self.L))
        for j in range(self.L):
            Y[:,j] = self.h[j].predict(X)
        return Y

    def predict_proba(self, X):
        '''
            return confidence predictions for X
            WARNING: returns a multi-label (binary) distribution only at the moment.
        '''
        N,D = X.shape
        P = zeros((N,self.L))
        for j in range(self.L):
            P[:,j] = self.h[j].predict_proba(X)[:,1]
        return P

    def get_info(self):
        return 'estimator'

    def get_class_type(self):
        pass

    def score(self, X, y):
        pass

def demo():
    import sys
    sys.path.append( '../data' )
    from skmultiflow.data.synth import make_logical

    X,Y = make_logical()
    N,L = Y.shape

    h = MultiOutputLearner(linear_model.SGDClassifier(n_iter=100))
    h.fit(X, Y)

    p = h.predict(X)
    ham = hamming_score(Y, p)
    print(ham)
    # Test
    print(h.predict(X))
    print("vs")
    print(Y)

if __name__ == '__main__':
    demo()

