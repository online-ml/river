__author__ = 'Guilherme Matsumoto'

import numpy as np
import copy as cp
from skmultiflow.core.base_object import BaseObject
from skmultiflow.classification.lazy.neighbours.distances import mixed_distance, euclidean_distance
from skmultiflow.core.utils.utils import *


class KDTree(BaseObject):
    """ Simplistic kd tree implementation

        For the purpose of the KNN algorithm, there is no need of adding and removing elements from the tree, so they 
        are not currently implemented.

        Accepts normal integer coded categorical features. If X contains one-hot encoded features use a pipeline with 
        a one_hot_to_categorical transform.

        Robust for mixed categorical and numerical X matrix.

        The left subtree goes up to, but does not include, the splitting_value. The right subtree starts from, and 
        includes, the splitting value and goes to the end of the samples 
    """
    METRICS = ['mixed', 'euclidean']

    def __init__(self, X, metric='mixed', categorical_list=None, return_distance=False, leaf_size=40, **kwargs):
        """ KDTree constructor

        :param X: Features matrix. shape: (n_samples, n_features). 
        :param metric: The distance metric to be used. Needs to implement BaseDistanceMetric
        :param categorical_list: List of lists with all categorical features. If a categorical attribute is one-hot 
                                 encoded its associated list should contain all indexes of the one-hot coding
        :param kwargs: Additional keyword arguments are passed to the metric function
        """

        super().__init__()

        self.distance_function = None

        if metric not in self.METRICS:
            raise ValueError("The metric '" + metric + "' is not supported by the KDTree.")

        if metric == self.METRICS[0]:
            self.distance_function = mixed_distance
        if metric == self.METRICS[1]:
            self.distance_function = euclidean_distance

        self.X = np.asarray(X)
        if self.X.ndim != 2:
            raise ValueError("X should be a matrix, or array-like, of shape (n_samples, n_features).")

        self.n_samples, self.n_features = self.X.shape

        self.categorical_list = categorical_list
        self.return_distance = return_distance
        self.leaf_size = leaf_size

        self.kwargs = kwargs if kwargs is not None else {}

        distance_array = [0.0 for i in range(self.n_features)]
        for i in range(self.n_features):
            if self.categorical_list is not None:
                if i not in self.categorical_list:
                    distance_array[i] = max(self.X[:, i].flatten()) - min(self.X[:, i].flatten())
            else:
                distance_array[i] = max(self.X[:, i].flatten()) - min(self.X[:, i].flatten())

        self.kwargs['distance_array'] = distance_array
        self.kwargs['categorical_list'] = self.categorical_list
        self.root = None
        self.maxes = None
        self.mins = None
        self.create_tree()

    def create_tree(self):
        # Getting mins and maxes from features
        self.maxes = np.amax(self.X, axis=0)
        self.mins = np.amin(self.X, axis=0)

        # Getting de splitting dimension
        d = np.argmax(self.maxes - self.mins)

        maxval = self.maxes[d]
        minval = self.mins[d]

        data = self.X[:, d]
        split = (maxval+minval)/2
        left = np.nonzero(data < split)[0]
        right = np.nonzero(data >= split)[0]
        if (len(left) == 0):
            split = np.amin(data[data != np.amin(data)])
            left = np.nonzero(data < split)[0]
            right = np.nonzero(data >= split)[0]
        if (len(right) == 0):
            split = np.amax(data[data != np.amax(data)])
            left = np.nonzero(data < split)[0]
            right = np.nonzero(data >= split)[0]

        # Create the root node, which recursively creates all the other nodes
        self.root = KDTreeNode(data=self.X, left_indexes=left, right_indexes=right,
                               split_axis=d, split_value=split, distance_function=self.distance_function,
                               leaf_size=self.leaf_size, **self.kwargs)

    def query(self, X, k=1):
        """ Searches the tree for the k nearest neighbors of X

        :param X: Array-like of size n_features
        :param k: The number of nearest neighbors to query for
        :return: The k nearest neighbors of sample X
        """
        r, c = get_dimensions(X)
        dist_all, ind_all = [], []

        if (r == 1) and ((not isinstance(X[0], type([]))) and (not isinstance(X[0], type(np.ndarray([0]))))):
            neighbors_distances = []
            neighbors_distances = self.root.query_node(X, k, neighbors_distances)
            indexes, distances = [], []
            for key, value in neighbors_distances:
                indexes.append(key)
                distances.append(value)
            dist_all.append(distances)
            ind_all.append(indexes)
            if self.return_distance:
                return dist_all, ind_all
            else:
                return ind_all
        else:
            for i in range(r):
                neighbors_distances = []
                neighbors_distances = self.root.query_node(X[i], k, neighbors_distances)
                indexes, distances = [], []
                for key, value in neighbors_distances:
                    indexes.append(key)
                    distances.append(value)
                dist_all.append(distances)
                ind_all.append(indexes)
            if self.return_distance:
                return dist_all, ind_all
            else:
                return ind_all

    def query_radius(self, X, r):
        pass

    @property
    def _root(self):
        return self.root

    def get_info(self):
        return 'Not implemented.'

    def get_class_type(self):
        return 'data_structure'


class KDTreeNode(BaseObject):
    def __init__(self, data, left_indexes, right_indexes, split_axis, split_value, distance_function, leaf_size,
                 **kwargs):
        super().__init__()

        self.data = data

        self.left_subtree = None
        self.right_subtree = None
        self.left_indexes = left_indexes
        self.right_indexes = right_indexes

        self.split_axis = split_axis
        self.split_value = split_value
        self.leaf_size = leaf_size
        self.is_leaf = False
        self.leaf_indexes = None

        # Here we assume left and right have the same dimensions.
        self.split_axis = split_axis

        self.distance_function = distance_function
        self.kwargs = kwargs

        self._start_node()

    def _start_node(self):
        # Handling left subtree
        sum = 0
        if self.left_indexes is not None:
            sum += len(self.left_indexes)
        if self.right_indexes is not None:
            sum += len(self.right_indexes)
        if sum <= self.leaf_size:
            self.is_leaf = True
            self.left_subtree = None
            self.right_subtree = None
            if self.left_indexes is None and self.right_indexes is not None:
                self.leaf_indexes = self.right_indexes
            elif self.left_indexes is not None and self.right_indexes is None:
                self.leaf_indexes = self.left_indexes
            else:
                self.leaf_indexes = list(set().union(self.left_indexes, self.right_indexes))

        else:

            if self.left_indexes is not None:
                if len(self.left_indexes) > 0:
                    aux_X = np.asarray([self.data[index] for index in self.left_indexes])
                    maxes = np.amax(aux_X, axis=0)
                    mins = np.amin(aux_X, axis=0)

                    # Getting longest side
                    d = np.argmax(maxes - mins)

                    maxval = maxes[d]
                    minval = mins[d]

                    data = aux_X[:, d]

                    # Get the split point and the current root node
                    split = (maxval + minval) / 2

                    left = np.nonzero(data < split)[0]
                    left = np.asarray([self.left_indexes[k] for k in left])
                    right = np.nonzero(data >= split)[0]
                    right = np.asarray([self.left_indexes[k] for k in right])

                    if (len(right) == 0) or (len(left) == 0):
                        if (len(right) == 0) and (len(left) > self.leaf_size):
                            split = np.amax(data[data != np.amax(data)])
                            left = np.nonzero(data < split)[0]
                            left = np.asarray([self.left_indexes[k] for k in left])
                            right = np.nonzero(data >= split)[0]
                            right = np.asarray([self.left_indexes[k] for k in right])
                        elif (len(right) == 0) and (len(left) <= self.leaf_size):
                            right = None
                        elif (len(left) == 0) and (len(right) > self.leaf_size):
                            split = np.amin(data[data != np.amin(data)])
                            left = np.nonzero(data < split)[0]
                            left = np.asarray([self.left_indexes[k] for k in left])
                            right = np.nonzero(data >= split)[0]
                            right = np.asarray([self.left_indexes[k] for k in right])
                        elif (len(left) == 0) and (len(right) <= self.leaf_size):
                            left = None

                    # Creating left subtree
                    self.left_subtree = KDTreeNode(data=self.data, left_indexes=left, right_indexes=right, split_axis=d,
                                                   split_value=split, distance_function=self.distance_function,
                                                   leaf_size=self.leaf_size, **self.kwargs)

                else:
                    self.left_subtree = None
            else:
                self.left_subtree = None

            # Handling right subtree
            if self.right_indexes is not None:
                if len(self.right_indexes) > 0:
                    aux_X = np.asarray([self.data[index] for index in self.right_indexes])
                    maxes = np.amax(aux_X, axis=0)
                    mins = np.amin(aux_X, axis=0)

                    # Getting longest side
                    d = np.argmax(maxes - mins)

                    maxval = maxes[d]
                    minval = mins[d]

                    data = aux_X[:, d]

                    # Get the split point and the current root node
                    split = (maxval + minval) / 2

                    left = np.nonzero(data < split)[0]
                    left = np.asarray([self.right_indexes[k] for k in left])
                    right = np.nonzero(data >= split)[0]
                    right = np.asarray([self.right_indexes[k] for k in right])

                    if (len(right) == 0) or (len(left) == 0):
                        if (len(right) == 0) and (len(left) > self.leaf_size):
                            split = np.amax(data[data != np.amax(data)])
                            left = np.nonzero(data < split)[0]
                            left = np.asarray([self.right_indexes[k] for k in left])
                            right = np.nonzero(data >= split)[0]
                            right = np.asarray([self.right_indexes[k] for k in right])
                        elif (len(right) == 0) and (len(left) <= self.leaf_size):
                            right = None
                        elif (len(left) == 0) and (len(right) > self.leaf_size):
                            split = np.amin(data[data != np.amin(data)])
                            left = np.nonzero(data < split)[0]
                            left = np.asarray([self.right_indexes[k] for k in left])
                            right = np.nonzero(data >= split)[0]
                            right = np.asarray([self.right_indexes[k] for k in right])
                        elif (len(left) == 0) and (len(right) <= self.leaf_size):
                            left = None

                    # Creating left subtree
                    self.right_subtree = KDTreeNode(data=self.data, left_indexes=left, right_indexes=right, split_axis=d,
                                                    split_value=split, distance_function=self.distance_function,
                                                    leaf_size=self.leaf_size, **self.kwargs)

                else:
                    self.right_subtree = None
            else:
                self.right_subtree = None

    def query_node(self, X, k, neighbors_distance_list):
        """ Queries a node and all of it's sub nodes, if there is a chance of finding a nearest neighbor in that branch. 

        :param X: Array-like containing the sample.
        :param k: Number of nearest neighbors to query for.
        :param neighbors_distance_list: A list of tuples of the form (index, distance), containing all the nodes that
        are already candidates to being a nearest neighbor

        :return: No return, but alters the neighbors_distance_list tuple list. 
        """
        # In case there is no more subtrees
        if not self:
            return neighbors_distance_list

        if self.is_leaf:
            for i in range(len(self.leaf_indexes)):
                dist = self.distance_function(instance_one=self.data[self.leaf_indexes[i]], instance_two=X, **self.kwargs)
                switch = 0
                while True:
                    if len(neighbors_distance_list) == 0:
                        switch = 0
                        break
                    else:
                        if switch >= len(neighbors_distance_list):
                            break
                        else:
                            if dist < neighbors_distance_list[switch][1]:
                                switch += 1
                            else:
                                #switch -= 1
                                break
                if switch > -1:
                    if len(neighbors_distance_list) < k:
                        # Add because the are not k elements in the list
                        neighbors_distance_list.insert(switch, (self.leaf_indexes[i], dist))
                    else:
                        # Add because the distance was in fact smaller than at least one distance already found
                        neighbors_distance_list.insert(switch, (self.leaf_indexes[i], dist))
                        del neighbors_distance_list[0]
                else:
                    if len(neighbors_distance_list) < k:
                        # Add because the list doesn't have k elements yet
                        neighbors_distance_list.insert(0, (self.leaf_indexes[i], dist))

        # If not a leaf enter here
        else:
            # Advance in the tree structure
            if X[self.split_axis] < self.split_value:
                if self.left_subtree is not None:
                    neighbors_distance_list = self.left_subtree.query_node(X, k, neighbors_distance_list)
            else:
                if self.right_subtree is not None:
                    neighbors_distance_list = self.right_subtree.query_node(X, k, neighbors_distance_list)

            # Check the other branch
            if X[self.split_axis] < self.split_value:
                if self.right_subtree is not None:
                    if len(neighbors_distance_list) < k:
                        neighbors_distance_list = self.right_subtree.query_node(X, k, neighbors_distance_list)
                    else:
                        aux = cp.deepcopy(self.kwargs)
                        aux['index'] = self.split_axis
                        dist = self.distance_function(X, self.split_value, **aux)
                        del aux
                        if dist < neighbors_distance_list[0][1]:
                            neighbors_distance_list = self.right_subtree.query_node(X, k, neighbors_distance_list)

            else:
                if self.left_subtree is not None:
                    if len(neighbors_distance_list) < k:
                        neighbors_distance_list = self.left_subtree.query_node(X, k, neighbors_distance_list)
                    else:
                        aux = cp.deepcopy(self.kwargs)
                        aux['index'] = self.split_axis
                        dist = self.distance_function(X, self.split_value, **aux)
                        del aux
                        if dist < neighbors_distance_list[0][1]:
                            neighbors_distance_list = self.left_subtree.query_node(X, k, neighbors_distance_list)

        return neighbors_distance_list

    @property
    def _is_leaf(self):
        return self.is_leaf

    @property
    def _left(self):
        return self.left_subtree

    @property
    def _right(self):
        return self.right_subtree

    def get_info(self):
        return 'Not implemented.'

    def get_class_type(self):
        return 'data_structure'