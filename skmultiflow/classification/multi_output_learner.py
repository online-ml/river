__author__ = 'Jesse Read'

import copy as cp
from numpy import *
from sklearn import linear_model
from skmultiflow.classification.base import BaseClassifier
from skmultiflow.evaluation.metrics.metrics import *


class MultiOutputLearner(BaseClassifier) :
    ''' MultiOutputLearner
    
    A Meta Learner. Does either classifier or regression, depending on 
    base learner (which by default is LogisticRegression). It will keep 
    one instance of the base learner for each classification task, in 
    a way that each classifier is in charge of a single classification 
    problem.
    
    Should be used to make single output predictors capable of learning 
    a multi output problem.
    
    Parameters
    ----------
    h: classifier (extension of the BaseClassifier)
        This is the ensemble classifier type, each ensemble classifier is going 
        to be a copy of the h classifier.
    
    '''

    h = None
    L = -1

    def __init__(self, h=linear_model.SGDClassifier(n_iter=100)):
        super().__init__()
        self.hop = h
        self.h = None
        self.L = None

    def __configure(self):
        self.h = [cp.deepcopy(self.hop) for j in range(self.L)]

    def fit(self, X, Y, classes = None):
        """ fit

        Fit the N classifiers, one for each classification task.

        Parameters
        ----------
        X : Numpy.ndarray of shape (n_samples, n_features)
            The array of samples used to fit the model.

        y: Array-like
            An array-like with the labels of all samples in X.

        classes: Array-like
            Optional parameter that contains all labels that may appear 
            in samples.

        Returns
        -------
        self

        """
        N,L = Y.shape
        self.L = L
        self.h = [cp.deepcopy(self.hop) for j in range(self.L)]

        for j in range(self.L):
            self.h[j].fit(X, Y[:,j])
        return self

    def partial_fit(self, X, Y, classes=None):
        """ partial_fit

        Partially fit each of the classifiers on the X matrix and the 
        corresponding entry at the Y array.

        Parameters
        ----------
        X : Numpy.ndarray of shape (n_samples, n_features)
            The array of samples used to fit the model.

        y: Array-like
            An array-like with the labels of all samples in X.

        classes: Array-like
            Contains all labels that may appear in samples. It's an optional 
            parameter, except during the first partial_fit call, when it's 
            obligatory.

        Returns
        -------
        self

        """
        N,self.L = Y.shape

        if self.h is None:
            self.__configure()

        for j in range(self.L):
            self.h[j].partial_fit(X, Y[:,j], classes)

        return self

    def predict(self, X):
        ''' predict
            
        Iterates over all the classifiers, predicting with each one, to obtain 
        the multi output prediction.
        
        Parameters
        ----------
        X : Numpy.ndarray of shape (n_samples, n_features)
            The matrix of samples one wants to predict.
            
        Returns
        -------
        An array-like with all the predictions for the samples in X.
        
        '''
        N,D = X.shape
        Y = zeros((N,self.L))
        for j in range(self.L):
            Y[:,j] = self.h[j].predict(X)
        return Y

    def predict_proba(self, X):
        """ predict_proba
        
        Estimates the probability of each sample in X belonging to each of 
        the existing labels for each of the classification tasks.
        
        It's a simple call to all of the classifier's predict_proba function, 
        return the probabilities for all the classification problems.

        Parameters
        ----------
        X : Numpy.ndarray of shape (n_samples, n_features)
            The matrix of samples one wants to predict.

        Returns
        -------
        An array of shape (n_samples, n_classification_tasks, n_labels), in which 
        we store the probability that each sample in X belongs to each of the labels, 
        in each of the classification tasks.
        
        """
        N,D = X.shape
        P = zeros((N,self.L))
        for j in range(self.L):
            P[:,j] = self.h[j].predict_proba(X)[:,1]
        return P

    def get_info(self):
        return ''


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

