__author__ = 'Guilherme Matsumoto'

import sys
from skmultiflow.core.base_object import BaseObject
import numpy as np


class FastBuffer(BaseObject):
    """ Keeps simple, unitary objects, in a limited size buffer.
        
    """
    def __init__(self, max_size, object_list=None):
        super().__init__()
        #default values
        self.current_size = 0
        self.max_size = None
        self.buffer = []

        self.configure(max_size, object_list)

    def get_class_type(self):
        return 'data_structure'

    def configure(self, max_size, object_list):
        self.max_size = max_size
        if object_list is not None:
            self.buffer = object_list

    def add_element(self, element_list):
        #print(element_list)
        if (self.current_size+len(element_list)) <= self.max_size:
            for i in range(len(element_list)):
                self.buffer.append(element_list[i])
                #print(element_list[i])
            self.current_size += len(element_list)
            return None
        else:
            aux = []
            #for i in range(len(element_list)):
            if self.isfull():
                aux.append(self.get_next_element())
            else:
                self.current_size += 1
            self.buffer.append(element_list[0])
            if len(element_list) > 1:
                aux.extend(self.add_element(element_list[1:]))
            return aux

    def get_next_element(self):
        return self.buffer.pop(0)

    def clear_queue(self):
        self._clear_all()

    def _clear_all(self):
        del self.buffer[:]

    def print_queue(self):
        print(self.buffer)

    def isfull(self):
        return self.current_size == self.max_size

    def isempty(self):
        return self.current_size == 0

    def get_current_size(self):
        return self.current_size

    def peek(self):
        try:
            return self.buffer[0]
        except IndexError:
            return None

    def get_queue(self):
        return self.buffer

    def get_info(self):
        return 'Not implemented.'


class FastComplexBuffer(BaseObject):
    """ Keeps a limited size buffer with predictions for a n number of targets
    
    """
    def __init__(self, max_size, width):
        super().__init__()
        #default values
        self.current_size = 0
        self.max_size = None
        self.width = None
        self.buffer = []

        self.configure(max_size, width)

    def get_class_type(self):
        return 'data_structure'

    def configure(self, max_size, width):
        self.max_size = max_size
        self.width = width

    def add_element(self, element_list):
        is_list = True
        dim = 1
        if hasattr(element_list, 'ndim'):
            dim = element_list.ndim
        if (dim > 1) or hasattr(element_list[0], 'append'):
            size, width = 0, 0
            if hasattr(element_list, 'append'):
                size, width = len(element_list), len(element_list[0])
            elif hasattr(element_list, 'shape'):
                is_list = False
                size, width = element_list.shape
            self.width = width
            if width != self.width:
                return None
        else:
            size, width = 0, 0
            if hasattr(element_list, 'append'):
                size, width = 1, len(element_list)
            elif hasattr(element_list, 'size'):
                is_list = False
                size, width = 1, element_list.size
            self.width = width
            if width != self.width:
                return None

        if not is_list:
            if size == 1:
                list = [element_list.tolist()]
            else:
                list = element_list.tolist()
        else:
            if size == 1:
                list = [element_list]
            else:
                list = element_list

        if (self.current_size+size) <= self.max_size:
            for i in range(size):
                self.buffer.append(list[i])
            self.current_size += size
            return None
        else:
            aux = []
            if self.isfull():
                aux.append(self.get_next_element())
            else:
                self.current_size += 1
            self.buffer.append(list[0])
            if len(list) > 1:
                aux.extend(self.add_element(list[1:]))
            return aux

    def get_next_element(self):
        return self.buffer.pop(0)

    def clear_queue(self):
        self._clear_all()

    def _clear_all(self):
        del self.buffer[:]

    def print_queue(self):
        print(self.buffer)

    def isfull(self):
        return self.current_size == self.max_size

    def isempty(self):
        return self.current_size == 0

    def get_current_size(self):
        return self.current_size

    def peek(self):
        try:
            return self.buffer[0]
        except IndexError:
            return None

    def get_queue(self):
        return self.buffer

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
            #print("1")
            return False
        m, n = self.confusion_matrix.shape
        if (i <= m) and (i >= 0) and (j <= n) and (j >= 0):
            return self._remove(i, j)
        else:
            #print("2")
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

    def get_sum_main_diagonal(self):
        m, n = self.confusion_matrix.shape
        sum = 0
        for i in range(m):
            sum += self.confusion_matrix[i, i]
        return sum

    @property
    def _matrix(self):
        if self.confusion_matrix is not None:
            return self.confusion_matrix
        else:
            return None

    @property
    def _sample_count(self):
        return self.sample_count

    def get_info(self):
        return 'Not implemented.'

    def get_class_type(self):
        return 'collection'


class MOLConfusionMatrix(BaseObject):
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
        self.dtype = dtype

        self.confusion_matrix = np.zeros((self.n_targets, 2, 2), dtype=dtype)

    def restart(self, n_targets):
        if n_targets is None:
            self.n_targets = 0
        else:
            self.n_targets = n_targets
        self.confusion_matrix = np.zeros((self.n_targets, 2, 2), dtype=self.dtype)
        pass

    def _update(self, target, true, pred):
        self.confusion_matrix[int(target), int(true), int(pred)] += 1
        return True

    def update(self, target=None, true = None, pred = None):
        if target is None or true is None or pred is None:
            return False
        else:
            m, n, p = self.confusion_matrix.shape
            if (target < m) and (target >= 0) and (true < n) and (true >= 0) and (pred < p) and (pred >= 0):
                return self._update(target, true, pred)
            else:
                if (true > 1) or (true < 0) or (pred > 1) or (pred < 0):
                    return False
                if target > m:
                    return False
                else:
                    self.reshape(target+1, 2, 2)
                    return self._update(target, true, pred)

    def remove(self, target=None, true = None, pred = None):
        if true is None or pred is None or target is None:
            return False
        m, n, p = self.confusion_matrix.shape
        if (target <= m) and (target >= 0) and (true <= n) and (true >= 0) and (pred >= 0) and (pred <= p):
            return self._remove(target, true, pred)
        else:
            return False

    def _remove(self, target, true, pred):
        self.confusion_matrix[target, true, pred] = self.confusion_matrix[target, true, pred] - 1
        return True

    def reshape(self, target, m, n):
        t, i, j = self.confusion_matrix.shape
        if (target > t+1) or (m != n) or (m != 2) or (m < i) or (n < j):
            return False
        aux = self.confusion_matrix.copy()
        self.confusion_matrix = np.zeros((target, m, n), self.dtype)
        for w in range(t):
            for p in range(i):
                for q in range(j):
                    self.confusion_matrix[w, p, q] = aux[w, p, q]
        return True

    def shape(self):
        return self.confusion_matrix.shape

    def value_at(self, target, i, j):
        return self.confusion_matrix[target, i, j]

    def row(self, r):
        return self.confusion_matrix[r:r+1, :]

    def column(self, c):
        return self.confusion_matrix[:, c:c+1]

    def target(self, t):
        return self.confusion_matrix[t, :, :]

    def get_sum_main_diagonal(self):
        t, m, n = self.confusion_matrix.shape
        sum = 0
        for i in range(t):
            sum += self.confusion_matrix[i, 0, 0]
            sum += self.confusion_matrix[i, 1, 1]
        return sum

    def get_total_sum(self):
        return np.sum(self.confusion_matrix)

    def get_total_discordance(self):
        return self.get_total_sum() - self.get_sum_main_diagonal()

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

class InstanceWindow(BaseObject):
    def __init__(self, num_attributes=0, num_target_tasks=1, categorical_list=None, max_size=1000, dtype=float):
        super().__init__()
        # default values
        self.dtype = None
        self.buffer = None
        self.n_attributes = None
        self.max_size = None
        self.categorical_attributes = None
        self.n_samples = None
        self.n_target_tasks = None
        self.configure(num_attributes=num_attributes, num_target_tasks=num_target_tasks, categorical_list=categorical_list, max_size=max_size, dtype=dtype)

    def configure(self, num_attributes, num_target_tasks, categorical_list=None, max_size=1000, dtype=float):
        self.n_attributes = num_attributes
        self.categorical_attributes = categorical_list
        self.max_size = max_size
        self.dtype = dtype
        self.n_target_tasks = num_target_tasks
        self.buffer = np.zeros((0, num_attributes+num_target_tasks))
        self.n_samples = 0

    def add_element(self, X, y):
        if (self.n_attributes != X.size):
            if self.n_samples == 0:
                self.n_attributes = X.size
                self.n_target_tasks = y.size
                self.buffer = np.zeros((0, self.n_attributes+self.n_target_tasks))
            else:
                raise ValueError("Number of attributes in X is different from the objects buffer dimension. "
                                 "Call configure() to correctly set up the InstanceWindow")

        if self.n_samples >= self.max_size:
            self.n_samples -= 1
            self.buffer = np.delete(self.buffer, 0, axis=0)

        if self.buffer is None:
            raise TypeError("None type not supported as the buffer, call configure() to correctly set up the InstanceWindow")

        aux = np.concatenate((X, y), axis=1)
        self.buffer = np.concatenate((self.buffer, aux), axis=0)
        self.n_samples += 1

    def delete_element(self):
        self.n_samples -= 1
        self.buffer = self.buffer[1:, :]
        pass

    def get_attributes_matrix(self):
        return self.buffer[:, :self.n_attributes]

    def get_targets_matrix(self):
        return self.buffer[:, self.n_attributes:]

    def at_index(self, index):
        return self.get_attributes_matrix()[index], self.get_targets_matrix()[index]

    @property
    def _buffer(self):
        return self.buffer

    @property
    def _num_target_tasks(self):
        return self.n_target_tasks

    @property
    def _num_attributes(self):
        return self.n_attributes

    @property
    def _num_samples(self):
        return self.n_samples

    def get_class_type(self):
        return 'data_structure'

    def get_info(self):
        return 'Not implemented.'

if __name__ == '__main__':
    text = '/asddfdsd/'
    aux = text.split("/")
    print(aux)


