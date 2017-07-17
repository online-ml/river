__author__ = 'Guilherme Matsumoto'

import numpy as np
from skmultiflow.classification.base import BaseClassifier
from skmultiflow.classification.core.driftdetection.adwin import ADWIN
#from skmultiflow.classification.lazy.neighbours.distances import euclidean_distance
from skmultiflow.core.utils.data_structures import InstanceWindow
from sklearn.neighbors import KDTree, DistanceMetric
from skmultiflow.classification.lazy.neighbours.distances import custom_distance


class KNN(BaseClassifier):
    def __init__(self, k=5, max_window_size=1000, leaf_size=30, categorical_list=[]):
        super().__init__()
        self.k = k
        self.max_window_size = max_window_size
        self.c = 0
        self.window = InstanceWindow(max_size=max_window_size, dtype=float)
        self.first_fit = True
        self.classes = []
        self.leaf_size = leaf_size
        self.categorical_list = categorical_list

    def fit(self, X, y, classes=None):
        r, c = 1, 1
        if isinstance(X, type([])):
            if isinstance(X[0], type([])):
                r, c = len(X), len(X[0])
            else:
                c = len(X)
        elif isinstance(X, type(np.array([0]))):
            if hasattr(X, 'shape'):
                r, c = X.shape
            elif hasattr(X, 'size'):
                r, c = 1, X.size

        if self.window is None:
            self.window = InstanceWindow(max_size=self.max_window_size)

        for i in range(r):
            if r > 1:
                self.window.add_element(np.asarray([X[i]]), np.asarray([[y[i]]]))
            else:
                self.window.add_element(np.asarray([X[i]]), np.asarray([[y[i]]]))
        return self

    def partial_fit(self, X, y, classes=None):
        raise NotImplementedError("The partial fit is not implemented for this module.")


    def predict(self, X):
        r, c = 1, 1
        if isinstance(X, type([])):
            if isinstance(X[0], type([])):
                r, c = len(X), len(X[0])
            else:
                c = len(X)
        elif isinstance(X, type(np.array([0]))):
            if X.ndim > 1:
                r, c = X.shape
            else:
                r, c = 1, X.size
            '''
            if hasattr(X, 'shape'):
                r, c = X.shape
            elif hasattr(X, 'size'):
                r,  c = 1, X.size
            '''

        aux, probs = self.predict_proba(X)
        preds = []
        for i in range(r):
            preds.append(aux[probs[i].index(np.max(probs[i]))])
        return preds

    def _predict(self, X):
        pass

    def predict_proba(self, X):
        """ Gives the class probability for the X sample (or X samples)
        
        :param X: 
        :return: return the list of classes and a list containing the probabilities of those classes 
        """
        probs = []
        r, c = 1, 1
        if isinstance(X, type([])):
            if isinstance(X[0], type([])):
                r, c = len(X), len(X[0])
            else:
                c = len(X)
        elif isinstance(X, type(np.array([0]))):
            if X.ndim > 1:
                r, c = X.shape
            else:
                r, c = 1, X.size
            '''
            if hasattr(X, 'shape'):
                r, c = X.shape
            elif hasattr(X, 'size'):
                r,  c = 1, X.size
            '''

        self.classes = list(set().union(self.classes, np.unique(self.window.get_targets_matrix())))

        classes = [0 for i in range(len(self.classes))]

        new_dist, new_ind = self._predict_proba(X)
        for i in range(r):
            classes = [0 for i in range(len(self.classes))]
            for index in new_ind[i]:
                classes[self.classes.index(self.window.get_targets_matrix()[index])] += 1
            probs.append([x/self.k for x in classes])

        '''
        for i in range(r):
            classes = [0 for i in range(len(self.classes))]
            new_dist, new_ind = self._predict_proba(X[i])
            print(new_dist)
            print(new_ind)
            for index in new_ind:
                classes[self.classes.index(self.window.get_targets_matrix()[index])] += 1

            probs.append([x/self.k for x in classes])
        '''
        return self.classes, probs

    def _predict_proba(self, X):
        tree = KDTree(self.window.get_attributes_matrix(),self.leaf_size,metric='euclidean')
        dist, ind = tree.query(np.asarray(X), k=self.k)
        return dist, ind

    def score(self, X, y):
        pass

    def get_info(self):
        return 'Not implemented.'