__author__ = 'Jesse Read'

from numpy import *
import copy
from sklearn import linear_model
from skmultiflow.classification.base import BaseClassifier

class MultiOutputLearner(BaseClassifier) :
    '''
        Multi-Output Learner
        --------------------
        A Meta Learner. Does either classifier or regression, depending on base learner 
        (which by default is LogisticRegression).
    '''

    h = None
    L = -1

    def __init__(self, h=linear_model.LogisticRegression()):
        super().__init__()
        self.hop = h

    def fit(self, X, Y, classes = None):
        N,L = Y.shape
        self.L = L
        self.h = [ copy.deepcopy(self.hop) for j in range(self.L)]

        for j in range(self.L):
            self.h[j].fit(X, Y[:,j])
        return self

    def partial_fit(self, X, Y, classes=None):
        N,self.L = Y.shape

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

def demo():
    import sys
    sys.path.append( '../data' )
    from synth import make_logical

    X,Y = make_logical()
    N,L = Y.shape

    h = MOL(linear_model.SGDClassifier(n_iter=100))
    h.fit(X, Y)

    # Test
    print(h.predict(X))
    print("vs")
    print(Y)

if __name__ == '__main__':
    demo()

