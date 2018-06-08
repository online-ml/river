import numpy as np
import copy
from sklearn.linear_model import SGDRegressor

class RegressorChain() :
    '''
        Regressor Chain

        See also: `classifier_chains.py`
    '''

    h = None
    L = -1

    def __init__(self, h=SGDRegressor(), order=None):
        ''' init

            Parameters
            ----------
            h : sklearn model
                The base regressor
            order : str
                None to use default order, 'random' for random order.
        '''
        self.base_model = h
        self.order = order

    def fit(self, X, Y):
        ''' fit
        '''
        N, self.L = Y.shape
        L = self.L
        N, D = X.shape

        self.chain = np.arange(L)
        if self.order == 'random':
            np.random.shuffle(self.chain)

        # Set the chain order
        Y = Y[:,self.chain]

        # Train
        self.h = [ copy.deepcopy(self.base_model) for j in range(L)]
        XY = np.zeros((N, D + L-1))
        XY[:,0:D] = X
        XY[:,D:] = Y[:,0:L-1]
        for j in range(self.L):
            self.h[j].fit(XY[:,0:D+j], Y[:,j])
        return self

    def partial_fit(self, X, Y):
        ''' partial_fit

            N.B. Assume that fit has already been called
            (i.e., this is more of an 'update')
        '''
        N, self.L = Y.shape
        L = self.L
        N, D = X.shape

        # Set the chain order
        Y = Y[:,self.chain]

        XY = np.zeros((N, D + L-1))
        XY[:,0:D] = X
        XY[:,D:] = Y[:,0:L-1]
        for j in range(L):
            self.h[j].partial_fit(XY[:,0:D+j], Y[:,j])

        return self

    def predict(self, X):
        ''' predict

            Returns predictions for X
        '''
        N,D = X.shape
        Y = np.zeros((N,self.L))
        for j in range(self.L):
            if j>0:
                X = np.column_stack([X, Y[:,j-1]])
            Y[:,j] = self.h[j].predict(X)

        # Unset the chain order (back to default)
        return Y[:,np.argsort(self.chain)]



