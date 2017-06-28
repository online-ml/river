__author__ = 'Guilherme Matsumoto'

import numpy as np
from scipy import stats
from skmultiflow.filtering.base_transform import BaseTransform
from skmultiflow.core.utils.data_structures import FastBuffer


class MissingValuesCleaner(BaseTransform):
    def __init__(self, missing_value=np.nan, strategy='zero', window_size=200, new_value=1):
        """ 
        
        :param missing_value: value or value list to be transformed
        :param strategy: one of 'zero', 'mean', 'median', 'mode', 'custom'
        :param window_size: int. sliding window size, required if strategy is mean, median, or mode
        :param new_value: custom value to put in the place of missing_value
        """
        super().__init__()
        #default_values
        self.missing_value = np.nan
        self.strategy = 'zero'
        self.window_size = 200
        self.window = None
        self.new_value = 1

        self.configure(missing_value, strategy, window_size, new_value)

    def configure(self, missing_value, strategy, window_size, new_value=1):
        if hasattr(missing_value, 'append'):
            self.missing_value = missing_value
        else:
            self.missing_value = [missing_value]
        self.strategy = strategy
        self.window_size = window_size
        self.new_value = new_value

        if strategy in ['mean', 'median', 'mode']:
            self.window = FastBuffer(max_size=window_size)

    def transform(self, X):
        X = np.array(X)
        for i in range(len(X)):
            for j in range(len(X[i])):
                if X[i][j] in self.missing_value:
                    #print(np.array(self.window.get_queue()))
                    X[i][j] = self._get_substitute(j)
        return X

    def _get_substitute(self, column_index):
        if self.strategy == 'zero':
            return 0
        elif self.strategy == 'mean':
            if not self.window.isempty():
                return np.mean(np.array(self.window.get_queue())[:, column_index:column_index+1])
            else:
                return self.new_value
        elif self.strategy == 'median':
            print(self.window.get_queue())
            if not self.window.isempty():
                print('go2')
                return np.median(np.array(self.window.get_queue())[:, column_index:column_index+1].flatten())
            else:
                return self.new_value
        elif self.strategy == 'mode':
            if not self.window.isempty():
                return stats.mode(np.array(self.window.get_queue())[:, column_index:column_index+1].flatten())
            else:
                return self.new_value
        elif self.strategy == 'custom':
            return self.new_value

    def partial_fit_transform(self, X, y=None):
        X = self.transform(X)
        if self.strategy in ['mean', 'median', 'mode']:
            self.window.add_element(X)

        return X

    def partial_fit(self, X, y=None):
        X = np.asarray(X)
        if self.strategy in ['mean', 'meadian', 'mode']:
            self.window.add_element(X)
        pass

    def get_info(self):
        pass


