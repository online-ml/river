__author__ = 'Guilherme Matsumoto'

import numpy as np
from skmultiflow.core.base_object import BaseObject
from skmultiflow.core.utils.data_structures import FastBuffer

class ClassificationMeasurements(BaseObject):
    """
        i -> true labels
        j -> predictions
    """
    def __init__(self, targets=None, dtype=np.int64):
        super().__init__()
        if targets is not None:
            self.n_targets = len(targets)
        else:
            self.n_targets = 0
        self.confusion_matrix = ConfusionMatrix(self.n_targets, dtype)
        self.last_class = None
        self.sample_count = 0
        self.targets = targets

    def reset(self, targets=None):
        if targets is not None:
            self.n_targets = len(targets)
        else:
            self.n_targets = 0
        self.confusion_matrix.restart(self.n_targets)
        pass

    def add_result(self, sample, prediction):
        true_y = self._get_target_index(sample, True)
        pred = self._get_target_index(prediction, True)
        self.confusion_matrix.update(true_y, pred)
        self.sample_count += 1
        pass

    def get_majority_class(self):
        """ Get the true majority class
        
        :return: 
        """
        if (self.n_targets is None) or (self.n_targets == 0):
           return False
        majority_class = 0
        max_prob = 0.0
        for i in range(self.n_targets):
            sum = 0.0
            for j in range(self.n_targets):
                sum += self.confusion_matrix.value_at(i, j)
            sum = sum / self.sample_count
            if sum > max_prob:
                max_prob = sum
                majority_class = i

        return majority_class

    def get_performance(self):
        sum = 0
        n, l = self.confusion_matrix.shape()
        for i in range(n):
            sum += self.confusion_matrix.value_at(i, i)
        return sum / self.sample_count

    def get_incorrectly_classified_ratio(self):
        return 1.0 - self.get_performance()

    def _get_target_index(self, target, add = False):
        if (self.targets is None) and add:
            self.targets = []
            self.targets.append(target)
            self.n_targets = len(self.targets)
            self.confusion_matrix.reshape(len(self.targets), len(self.targets))
        elif (self.targets is None) and (not add):
            return None
        if ((target not in self.targets) and (add)):
            self.targets.append(target)
            self.n_targets = len(self.targets)
            self.confusion_matrix.reshape(len(self.targets), len(self.targets))
        for i in range(len(self.targets)):
            if self.targets[i] == target:
                return i
        return None

    def get_kappa(self):
        p0 = self.get_performance()
        pc = 0.0
        n, l = self.confusion_matrix.shape()
        for i in range(n):
            row = self.confusion_matrix.row(i)
            column = self.confusion_matrix.column(i)

            sum_row = np.sum(row) / self.sample_count
            sum_column = np.sum(column) / self.sample_count

            pc += sum_row * sum_column

        return (p0 - pc) / (1.0 - pc)

    @property
    def _matrix(self):
        return self.confusion_matrix._matrix

    def get_info(self):
        return 'Not implemented.'

    def get_class_type(self):
        return 'collection'


class WindowClassificationMeasurements(BaseObject):
    def __init__(self, targets=None, dtype=np.int64, window_size=200):
        super().__init__()
        if targets is not None:
            self.n_targets = len(targets)
        else:
            self.n_targets = 0
        self.confusion_matrix = ConfusionMatrix(self.n_targets, dtype)
        self.last_class = None

        self.targets = targets
        self.window_size = window_size
        self.true_labels = FastBuffer(window_size)
        self.predictions = FastBuffer(window_size)
        self.temp = 0
        pass

    def reset(self, targets=None):
        if targets is not None:
            self.n_targets = len(targets)
        else:
            self.n_targets = 0
        self.confusion_matrix.restart(self.n_targets)
        pass

    def add_result(self, sample, prediction):
        true_y = self._get_target_index(sample, True)
        pred = self._get_target_index(prediction, True)

        old_true = self.true_labels.add_element(np.array([sample]))
        old_predict = self.predictions.add_element(np.array([prediction]))
        #print(str(old_true) + ' ' + str(old_predict))

        if (old_true is not None) and (old_predict is not None):
            self.temp += 1
            error = self.confusion_matrix.remove(self._get_target_index(old_true[0]), self._get_target_index(old_predict[0]))
            #if not error:
                #print("errou")

        self.confusion_matrix.update(true_y, pred)
        pass

    def get_majority_class(self):
        """ Get the true majority class

        :return: 
        """
        if (self.n_targets is None) or (self.n_targets == 0):
            return False
        majority_class = 0
        max_prob = 0.0
        for i in range(self.n_targets):
            sum = 0.0
            for j in range(self.n_targets):
                sum += self.confusion_matrix.value_at(i, j)
            sum = sum / self.true_labels.get_current_size()
            if sum > max_prob:
                max_prob = sum
                majority_class = i

        return majority_class

    def get_performance(self):
        sum = 0
        n, l = self.confusion_matrix.shape()
        for i in range(n):
            sum += self.confusion_matrix.value_at(i, i)
        return sum / self.true_labels.get_current_size()

    def get_incorrectly_classified_ratio(self):
        return 1.0 - self.get_performance()

    def _get_target_index(self, target, add=False):
        if (self.targets is None) and add:
            self.targets = []
            self.targets.append(target)
            self.n_targets = len(self.targets)
            self.confusion_matrix.reshape(len(self.targets), len(self.targets))
        elif (self.targets is None) and (not add):
            return None
        if ((target not in self.targets) and (add)):
            self.targets.append(target)
            self.n_targets = len(self.targets)
            self.confusion_matrix.reshape(len(self.targets), len(self.targets))
        for i in range(len(self.targets)):
            if self.targets[i] == target:
                return i
        return None

    def get_kappa(self):
        p0 = self.get_performance()
        pc = 0.0
        n, l = self.confusion_matrix.shape()
        for i in range(n):
            row = self.confusion_matrix.row(i)
            column = self.confusion_matrix.column(i)

            sum_row = np.sum(row) / self.true_labels.get_current_size()
            sum_column = np.sum(column) / self.true_labels.get_current_size()

            pc += sum_row * sum_column

        return (p0 - pc) / (1.0 - pc)

    @property
    def _matrix(self):
        return self.confusion_matrix._matrix

    @property
    def _sample_count(self):
        return self.true_labels.get_current_size()

    def get_class_type(self):
        return 'collection'

    def get_info(self):
        return 'Not implemented.'


class ConfusionMatrix(BaseObject):
    """
        i -> true_labels
        j -> predictions
    """
    def __init__(self, n_targets=None, dtype=np.int64):
        super().__init__()
        if n_targets is not None:
            self.n_targets = n_targets
        else:
            self.n_targets = 0
        self.sample_count = 0
        self.dtype = dtype

        self.confusion_matrix = np.zeros((self.n_targets, self.n_targets), dtype=dtype)

    def restart(self, n_targets):
        if n_targets is None:
            self.n_targets = 0
        else:
            self.n_targets = n_targets
        self.confusion_matrix = np.zeros((self.n_targets, self.n_targets))
        self.sample_count = 0
        pass

    def _update(self, i, j):
        self.confusion_matrix[i, j] += 1
        self.sample_count += 1
        return True

    def update(self, i = None, j = None):
        if i is None or j is None:
            return False
        else:
            m, n = self.confusion_matrix.shape
            if (i <= m) and (i >= 0) and (j <= n) and (j >= 0):
                return self._update(i, j)
            else:
                max = np.max(i, j)
                if max > m+1:
                    return False
                else:
                    self.reshape(max, max)
                    return self._update(i, j)

    def remove(self, i = None, j = None):
        if i is None or j is None:
            print("1")
            return False
        m, n = self.confusion_matrix.shape
        if (i <= m) and (i >= 0) and (j <= n) and (j >= 0):
            return self._remove(i, j)
        else:
            print("2")
            return False

    def _remove(self, i, j):
        self.confusion_matrix[i, j] = self.confusion_matrix[i, j] - 1
        self.sample_count -= 1
        return True

    def reshape(self, m, n):
        i, j = self.confusion_matrix.shape
        #print(self.confusion_matrix.shape)
        if (m != n) or (m < i) or (n < j):
            return False
        aux = self.confusion_matrix.copy()
        #print(aux)
        self.confusion_matrix = np.zeros((m, n), self.dtype)
        #print(self.confusion_matrix)
        for p in range(i):
            for q in range(j):
                self.confusion_matrix[p, q] = aux[p, q]
        return True

    def shape(self):
        return self.confusion_matrix.shape

    def value_at(self, i, j):
        return self.confusion_matrix[i, j]

    def row(self, r):
        return self.confusion_matrix[r:r+1, :]

    def column(self, c):
        return self.confusion_matrix[:, c:c+1]

    @property
    def _matrix(self):
        if self.confusion_matrix is not None:
            return self.confusion_matrix
        else:
            return None

    def get_info(self):
        return 'Not implemented.'

    def get_class_type(self):
        return 'collection'
