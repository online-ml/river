__author__ = 'Guilherme Matsumoto'

import numpy as np
import copy as cp
from skmultiflow.classification.base import BaseClassifier
from skmultiflow.classification.lazy.knn import KNN
from skmultiflow.classification.core.driftdetection.adwin import ADWIN
from skmultiflow.core.utils.utils import *


class LeverageBagging(BaseClassifier):
    LEVERAGE_ALGORITHMS = ['leveraging_bag', 'leveraging_bag_me', 'leveraging_bag_half', 'leveraging_bag_wt',
                           'leveraging_subag']
    def __init__(self, h=KNN(), ensemble_length=2, w=6, delta=0.002, enable_code_matrix=False, leverage_algorithm='leveraging_bag'):
        super().__init__()
        # default values
        self.h = h.reset()
        self.ensemble_length = None
        self.ensemble = None
        self.adwin_ensemble = None
        self.n_detected_changes = None
        self.matrix_codes = None
        self.enable_matrix_codes = None
        self.w = None
        self.delta = None
        self.classes = None
        self.leveraging_algorithm = None
        self.configure(h, ensemble_length, w, delta, enable_code_matrix, leverage_algorithm)
        self.init_matrix_codes = True

        self.adwin_ensemble = []
        for i in range(ensemble_length):
            self.adwin_ensemble.append(ADWIN(self.delta))

    def configure(self, h, ensemble_length, w, delta, enable_code_matrix, leverage_algorithm):
        self.ensemble_length = ensemble_length
        self.ensemble = [cp.deepcopy(h) for x in range(ensemble_length)]
        self.w = w
        self.delta = delta
        self.enable_matrix_codes = enable_code_matrix
        if leverage_algorithm not in self.LEVERAGE_ALGORITHMS:
            raise ValueError("Leverage algorithm not supported.")
        self.leveraging_algorithm = leverage_algorithm

    def fit(self, X, y, classes=None):
        raise NotImplementedError

    def partial_fit(self, X, y, classes=None):
        if classes is None and self.classes is None:
            raise ValueError("The first partial_fit call should pass all the classes.")
        if classes is not None and self.classes is None:
            self.classes = classes
        elif classes is not None and self.classes is not None:
            if self.classes == set(classes):
                pass
            else:
                raise ValueError(
                    "The classes passed to the partial_fit function differ from those passed in an earlier moment.")

        r, c = get_dimensions(X)
        for i in range(r):
            self._partial_fit(X[i], y[i])

    def _partial_fit(self, X, y):
        n_classes = len(self.classes)
        change = False

        if self.init_matrix_codes:
            self.matrix_codes = np.zeros((self.ensemble_length, len(self.classes)), dtype=int)
            for i in range(self.ensemble_length):
                n_zeros = 0
                n_ones = 0
                while((n_ones - n_zeros) * (n_ones - n_zeros) > self.ensemble_length % 2):
                    n_zeros = 0
                    n_ones = 0
                    for j in range(len(self.classes)):
                        result = 0
                        if (j == 1) and (len(self.classes) == 2):
                            result = 1 - self.matrix_codes[i][0]
                        else:
                            result = np.random.randint(2)

                        self.matrix_codes[i][j] = result
                        if result == 1:
                            n_ones += 1
                        else:
                            n_zeros += 1
            self.init_matrix_codes = False

        detected_change = False
        X_cp, y_cp = cp.deepcopy(X), cp.deepcopy(y)
        for i in range(self.ensemble_length):
            k = 0.0

            if self.leveraging_algorithm == self.LEVERAGE_ALGORITHMS[0]:
                k = np.random.poisson(self.w)

            elif self.leveraging_algorithm == self.LEVERAGE_ALGORITHMS[1]:
                error = self.adwin_ensemble[i]._estimation
                pred = self.ensemble[i].predict(np.asarray([X]))
                if pred is None:
                    k = 1.0
                elif pred[0] != y:
                    k = 1.0
                elif np.random.rand() < (error/(1.0 - error)):
                    k = 1.0
                else:
                    k = 0.0

            elif self.leveraging_algorithm == self.LEVERAGE_ALGORITHMS[2]:
                w = 1.0
                k = 0.0 if (np.random.randint(2) == 1) else w

            elif self.leveraging_algorithm == self.LEVERAGE_ALGORITHMS[3]:
                w = 1.0
                k = 1.0 + np.random.poisson(w)

            elif self.leveraging_algorithm == self.LEVERAGE_ALGORITHMS[4]:
                w = 1.0
                k = np.random.poisson(1)
                k = w if k > 0 else 0

            if k > 0:
                if self.enable_matrix_codes:
                    y_cp = self.matrix_codes[i][int(y_cp)]
                for l in range(int(k)):
                    self.ensemble[i].partial_fit(np.asarray([X_cp]), np.asarray([y_cp]), self.classes)

            try:
                pred = self.ensemble[i].predict(np.asarray([X]))
                if pred is not None:
                    add = 1 if (pred[0] == y_cp) else 0
                    error = self.adwin_ensemble[i]._estimation
                    self.adwin_ensemble[i].add_element(add)
                    if self.adwin_ensemble[i].detected_change():
                        if self.adwin_ensemble[i]._estimation > error:
                            change = True
            except ValueError:
                change = False

        if change:
            self.n_detected_changes += 1
            max = 0.0
            imax = -1
            for i in range(self.ensemble_length):
                if max < self.adwin_ensemble[i]._estimation:
                    max = self.adwin_ensemble[i]._estimation
                    imax = i
            if imax != -1:
                self.ensemble[imax].reset()
                self.adwin_ensemble[imax] = ADWIN(self.delta)
        return self

    def adjust_ensemble_size(self):
        if len(self.classes) != len(self.ensemble):
            if len(self.classes) > len(self.ensemble):
                for i in range(len(self.ensemble), len(self.classes)):
                    self.ensemble.append(cp.deepcopy(self.h))
                    self.adwin_ensemble.append(ADWIN(self.delta))
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
        if self.enable_matrix_codes:
            return self.predict_binary_proba(X)
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
        total_sum = np.sum(probs)
        aux = []
        for i in range(len(probs)):
            aux.append([x / total_sum for x in probs[i]])
        return aux

    def predict_binary_proba(self, X):
        probs = []
        r, c = get_dimensions(X)
        if not self.init_matrix_codes:
            try:
                for i in range(self.ensemble_length):
                    vote = self.ensemble[i].predict_proba(X)
                    vote_class = 0

                    if len(vote) == 2:
                        vote_class = 1 if (vote[1] > vote[0]) else 0

                    if len(probs) < 1:
                        for n in range(r):
                            probs.append([0.0 for x in vote[n]])

                    for j in range(len(self.classes)):
                        if self.matrix_codes[i][j] == vote_class:
                            probs[j] += 1
            except ValueError:
                return None

            if len(probs) < 1:
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
        return None

    def reset(self):
        self.configure(self.h, self.ensemble_length, self.w, self.delta, self.enable_matrix_codes)
        self.adwin_ensemble = []
        for i in range(self.ensemble_length):
            self.adwin_ensemble.append(ADWIN(self.delta))
        self.n_detected_changes = 0
        self.classes = None
        self.init_matrix_codes = True

    def score(self, X, y):
        raise NotImplementedError

    def get_info(self):
        return ''
