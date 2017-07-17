__author__ = 'Guilherme Matsumoto'

import sys
import numpy as np
from skmultiflow.classification.lazy.knn import KNN
from skmultiflow.classification.core.driftdetection.adwin import ADWIN
from skmultiflow.core.utils.data_structures import InstanceWindow


class KNNAdwin(KNN):
    def __init__(self, k=5, max_window_size=sys.maxsize, leaf_size=30, categorical_list=[]):
        super().__init__(k=k, max_window_size=max_window_size, leaf_size=leaf_size, categorical_list=categorical_list)
        self.adwin = ADWIN()
        self.window = None

    def partial_fit(self, X, y, classes=None):
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
            # self.window = InstanceWindow(max_size=2000)
            self.window = InstanceWindow(max_size=self.max_window_size)
        for i in range(r):
            self.window.add_element(np.asarray([X[i]]), np.asarray([[y[i]]]))

            if self.window._num_samples >= self.k:
                add = 1 if self.predict(X[0]) == y[i] else 0
                self.adwin.add_element(add)
            else:
                self.adwin.add_element(0)
        
        # size = self.adwin._width
        if self.window._num_samples >= self.k:
            changed = self.adwin.detected_change()
            if changed:

                # old = self.window._num_samples

                if self.adwin._width < self.window._num_samples:
                    for i in range(self.window._num_samples, self.adwin._width, -1):
                        self.window.delete_element()

                # print("\nold: " + str(size))
                # print("new: " + str(self.adwin._width))
                # print("old - window: " + str(old))
                # print("new - window: " + str(self.window._num_samples))


        return self

