__author__ = 'Guilherme Matsumoto'

import numpy as np
from skmultiflow.classification.base import BaseClassifier
from skmultiflow.classification.core.driftdetection.adwin import ADWIN
#from skmultiflow.classification.lazy.neighbours.distances import euclidean_distance
from skmultiflow.core.utils.data_structures import InstanceWindow
#from sklearn.neighbors import KDTree, DistanceMetric
from skmultiflow.classification.lazy.neighbours.kdtree import KDTree
import sklearn as sk
from skmultiflow.core.utils.utils import *
from skmultiflow.classification.lazy.neighbours.distances import custom_distance
from timeit import default_timer as timer


class KNN(BaseClassifier):
    """ K-Nearest Neighbours learner
    
        Not optimal for a mixture of categorical and numerical features.
    
    """
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
        r, c = get_dimensions(X)
        if self.window is None:
            self.window = InstanceWindow(max_size=self.max_window_size)

        for i in range(r):
            if r > 1:
                self.window.add_element(np.asarray([X[i]]), np.asarray([[y[i]]]))
            else:
                self.window.add_element(np.asarray([X[i]]), np.asarray([[y[i]]]))
        return self

    def partial_fit(self, X, y, classes=None):
        r, c = get_dimensions(X)
        if self.window is None:
            self.window = InstanceWindow(max_size=self.max_window_size)

        for i in range(r):
            if r > 1:
                self.window.add_element(np.asarray([X[i]]), np.asarray([[y[i]]]))
            else:
                self.window.add_element(np.asarray([X[i]]), np.asarray([[y[i]]]))
        return self

    def reset(self):
        self.window = None
        return self

    def predict(self, X):
        r, c = get_dimensions(X)
        '''
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

        '''
            if hasattr(X, 'shape'):
                r, c = X.shape
            elif hasattr(X, 'size'):
                r,  c = 1, X.size
        '''

        probs = self.predict_proba(X)
        preds = []
        for i in range(r):
            preds.append(self.classes[probs[i].index(np.max(probs[i]))])
        return preds

    def _predict(self, X):
        pass

    def predict_proba(self, X):
        """ Gives the class probability for the X sample (or X samples)
        
        :param X: 
        :return: return the list of classes and a list containing the probabilities of those classes 
        """
        if self.window is None:
            raise ValueError("KNN should be partially fitted on at least k samples before doing any prediction.")
        if self.window._num_samples < self.k:
            raise ValueError("KNN should be partially fitted on at least k samples before doing any prediction.")
        probs = []
        r, c = get_dimensions(X)
        '''
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

        '''
            if hasattr(X, 'shape'):
                r, c = X.shape
            elif hasattr(X, 'size'):
                r,  c = 1, X.size
        '''

        self.classes = list(set().union(self.classes, np.unique(self.window.get_targets_matrix())))

        new_dist, new_ind = self._predict_proba(X)
        for i in range(r):
            classes = [0 for i in range(len(self.classes))]
            for index in new_ind:
                classes[self.classes.index(self.window.get_targets_matrix()[index])] += 1
            probs.append([x/len(new_ind) for x in classes])

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
        # print(probs)
        return probs

    def _predict_proba(self, X):
        results = []

        start = timer()
        tree_aux = sk.neighbors.KDTree(self.window.get_attributes_matrix(),self.leaf_size,metric='euclidean')
        dist_aux, ind_aux = tree_aux.query(np.asarray(X), k=self.k)
        end = timer()
        tree = KDTree(self.window.get_attributes_matrix(), metric='modified_euclidean',
                      categorical_list=self.categorical_list, return_distance=True)
        dist, ind = tree.query(np.asarray(X), k=self.k)
        print("Create and query tree time: " + str(end-start))
        return dist, ind

    def score(self, X, y):
        pass

    def get_info(self):
        return 'Not implemented.'