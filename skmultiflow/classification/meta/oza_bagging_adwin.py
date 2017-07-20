__author__ = 'Guilherme Matsumoto'

import numpy as np
import copy as cp
from skmultiflow.classification.lazy.knn_adwin import KNNAdwin
from skmultiflow.classification.base import BaseClassifier
from skmultiflow.classification.core.driftdetection.adwin import ADWIN
from skmultiflow.core.utils.utils import *


class OzaBaggingAdwin(BaseClassifier):
    def __init__(self, h=KNNAdwin(), ensemble_length=2):
        super().__init__()
        # default values
        self.ensemble = None
        self.ensemble_length = None
        self.classes = None
        self.h = h.reset()
        self.configure(h, ensemble_length)

        self.adwin_ensemble = []
        for i in range(ensemble_length):
            self.adwin_ensemble.append(ADWIN())

    def configure(self, h, ensemble_length):
        self.ensemble_length = ensemble_length
        self.ensemble = [cp.deepcopy(h) for j in range(self.ensemble_length)]

    def fit(self, X, y, classes = None):
        raise NotImplementedError

    def partial_fit(self, X, y, classes=None):
        r, c = get_dimensions(X)
        if self.classes is None:
            if classes is None:
                raise ValueError("The first partial_fit call should pass all the classes.")
            else:
                self.classes = classes

        if self.classes is not None and classes is not None:
            if set(self.classes) == set(classes):
                pass
            else:
                raise ValueError(
                    "The classes passed to the partial_fit function differ from those passed in an earlier moment.")

        self.adjust_ensemble_size()
        change_detected = False
        for i in range(self.ensemble_length):
            k = np.random.poisson()
            if k > 0:
                for b in range(k):
                    self.ensemble[i].partial_fit(X, y, classes)

            try:
                pred = self.ensemble[i].predict(X)
                error_estimation = self.adwin_ensemble[i]._estimation
                for j in range(r):
                    if pred[j] is not None:
                        if pred[j] == y[j]:
                            self.adwin_ensemble[i].add_element(1)
                        else:
                            self.adwin_ensemble[i].add_element(0)
                if self.adwin_ensemble[i].detected_change():
                    if self.adwin_ensemble[i] > error_estimation:
                        change_detected = True
            except ValueError:
                change_detected = False
                pass

        if change_detected:
            max = 0.0
            imax = -1
            for i in range(self.ensemble_length):
                if max < self.adwin_ensemble[i]._estimation:
                    max = self.adwin_ensemble[i]._estimation
                    imax = i
            if imax != -1:
                self.ensemble[imax].reset()
                self.adwin_ensemble[imax] = ADWIN()

        return self

    def adjust_ensemble_size(self):
        if len(self.classes) != len(self.ensemble):
            if len(self.classes) > len(self.ensemble):
                for i in range(len(self.ensemble), len(self.classes)):
                    self.ensemble.append(cp.deepcopy(self.h))
                    self.adwin_ensemble.append(ADWIN())
                    self.ensemble_length += 1


    def predict(self, X):
        r, c = get_dimensions(X)
        probs = self.predict_proba(X)
        preds = []
        if probs is None:
            return None
        for i in range(r):
            preds.append(self.classes[probs[i].index(np.max(probs[i]))])
        return preds

    def predict_proba(self, X):
        """ Predicts the probability of classification

                :param X: Feature matrix. shape (n_samples, n_features)
                :return: A matrix with the probabilities for all classes of all samples passed in X. shape (n_samples, n_classes).
                        If the learners' ensemble requires a minimum number of partial_fit/fit calls pefore any prediction is 
                        made, this will return None
                """
        probs = []
        r, c = get_dimensions(X)
        try:
            for i in range(self.ensemble_length):
                partial_probs = self.ensemble[i].predict_proba(X)
                if len(partial_probs[0]) != len(self.classes):
                    raise ValueError(
                        "The number of classes is different in the bagging algorithm and in the chosen learning algorithm.")

                if len(probs) < 1:
                    for n in range(r):
                        probs.append([0.0 for x in partial_probs[n]])

                for n in range(r):
                    for l in range(len(partial_probs[n])):
                        probs[n][l] += partial_probs[n][l]
        except ValueError:
            return None

        # normalizing probabilities
        if r > 1:
            total_sum = []
            for l in range(r):
                total_sum.append(np.sum(probs[l]))
        else:
            total_sum = [np.sum(probs)]
        aux = []
        for i in range(len(probs)):
            aux.append([x / total_sum[i] for x in probs[i]])
        return aux

    def score(self, X, y):
        pass

    def reset(self):
        self.configure(self.h, self.ensemble_length)
        self.adwin_ensemble = []
        for i in range(self.ensemble_length):
            self.adwin_ensemble.append(ADWIN())

    def get_info(self):
        return ''