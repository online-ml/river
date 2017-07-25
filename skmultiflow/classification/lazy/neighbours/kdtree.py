__author__ = 'Guilherme Matsumoto'

import numpy as np
import copy as cp
from skmultiflow.core.base_object import BaseObject
from skmultiflow.classification.lazy.neighbours.distances import custom_distance
from skmultiflow.core.utils.utils import *
from skmultiflow.classification.lazy.neighbours.distances import custom_distance


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
    METRICS = ['modified_euclidean']
    def __init__(self, X, metric='modified_euclidean', categorical_list=None, return_distance=False, **kwargs):
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
            self.distance_function = custom_distance


        self.X = np.asarray(cp.deepcopy(X))
        if self.X.ndim != 2:
            raise ValueError("X should be a matrix, or array-like, of shape (n_samples, n_features).")

        self.n_samples, self.n_features = self.X.shape

        #if categorical_list is not None:
        #    self.n_features -= len(categorical_list)

        self.categorical_list = categorical_list
        self.return_distance = return_distance

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
        self.create_tree()

    def create_tree(self):
        axis = 0

        # Salculating base sample list, base index list and base median
        aux_list = list(cp.deepcopy(self.X))
        aux = sorted(range(len(aux_list)), key=lambda k: aux_list[k][axis])
        index_list = [i for i in range(len(aux_list))]
        index_list = [index_list[k] for k in aux]
        median = len(aux_list) // 2

        # Recalculating median to get the lowest indexed sample with the feature on axis 0 equals to the split_value
        new_median = -1
        for i in range(1, median + 1):
            if aux_list[median][axis] == aux_list[median-i][axis]:
                new_median = median - i
        median = new_median if new_median != -1 else median

        # The element that should be the root node
        current = index_list[median]

        # Splitting the sample list to be sent as parameter to the node creation. Won't be kept as to not waste memory
        left = index_list[:median]
        right = index_list[median+1:]

        # Releasing memory
        del aux_list

        # Create the root node, which recursively creates all the other nodes
        self.root = KDTreeNode(node_index=current, data=self.X, left_indexes=left, right_indexes=right,
                               split_axis=axis, split_value=self.X[median][axis],
                               distance_function=self.distance_function, **self.kwargs)

    def query(self, X, k=1):
        """ Searches the tree for the k nearest neighbors of X
        
        :param X: Array-like of size n_features
        :param k: The number of nearest neighbors to query for
        :return: The k nearest neighbors of sample X
        """
        neighbors_distances = []
        neighbors_distances = self.root.query_node(X, k, neighbors_distances)
        indexes, distances = [], []
        for key, value in neighbors_distances:
            indexes.append(key)
            distances.append(value)

        if self.return_distance:
            return distances, indexes
        else:
            return indexes

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
    def __init__(self, node_index, data, left_indexes, right_indexes, split_axis, split_value, distance_function, **kwargs):
        super().__init__()

        self.node_index = node_index
        self.data = data

        self.left_subtree = None
        self.right_subtree = None
        self.left_indexes = left_indexes
        self.right_indexes = right_indexes

        self.split_axis = split_axis
        self.split_value = split_value

        # Here we assume left and right have the same dimensions.
        r, c = get_dimensions(data)
        self.split_axis = split_axis % c

        self.distance_function = distance_function
        self.kwargs = kwargs

        self._start_node()

    def _start_node(self):
        axis = self.split_axis

        # Handling left subtree
        if self.left_indexes is not None:
            if len(self.left_indexes) > 0:

                axis = self.split_axis

                # Calculating base sample list, base index list and base median
                aux = [self.data[index] for index in self.left_indexes]
                aux_list = list(cp.deepcopy(aux))

                self.left_subtree = None
                aux = sorted(range(len(aux_list)), key=lambda k: aux_list[k][axis])
                index_list = [self.left_indexes[k] for k in aux]
                median = len(aux_list) // 2

                # Recalculating median to get the lowest indexed sample with the feature on axis 0 equals to the split_value
                new_median = -1
                for i in range(1, median + 1):
                    if aux_list[median][axis] == aux_list[median - i][axis]:
                        new_median = median - i
                median = new_median if new_median != -1 else median

                # The element that should be the root node
                current = index_list[median]

                # Splitting the sample list to be sent as parameter to the node creation. Won't be kept as to not waste memory
                left = index_list[:median]
                right = index_list[median + 1:]

                # Releasing memory
                del aux_list, aux

                # Creating left subtree
                self.left_subtree = KDTreeNode(node_index=current, data=self.data, left_indexes=left,
                                               right_indexes=right, split_axis=axis+1,
                                               split_value=self.data[median][axis],
                                               distance_function=self.distance_function, **self.kwargs)
            else:
                self.left_subtree = None
        else:
            self.left_subtree = None

        # Handling right subtree
        if self.right_indexes is not None:
            if len(self.right_indexes) > 0:

                axis = self.split_axis

                # Calculating base sample list, base index list and base median
                aux = [self.data[index] for index in self.right_indexes]
                aux_list = list(cp.deepcopy(aux))

                self.right_subtree = None
                aux = sorted(range(len(aux_list)), key=lambda k: aux_list[k][axis])
                index_list = [self.right_indexes[k] for k in aux]
                median = len(aux_list) // 2

                # Recalculating median to get the lowest indexed sample with the feature on axis 0 equals to the split_value
                new_median = -1
                for i in range(1, median + 1):
                    if aux_list[median][axis] == aux_list[median - i][axis]:
                        new_median = median - i
                median = new_median if new_median != -1 else median

                # The element that should be the root node
                current = index_list[median]

                # Splitting the sample list to be sent as parameter to the node creation. Won't be kept as to not waste memory
                left = index_list[:median]
                right = index_list[median + 1:]

                # Releasing memory
                del aux_list, aux

                # Creating left subtree
                self.right_subtree = KDTreeNode(node_index=current, data=self.data, left_indexes=left,
                                               right_indexes=right, split_axis=axis + 1,
                                               split_value=self.data[median][axis],
                                               distance_function=self.distance_function, **self.kwargs)

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

        X = X.flatten()
        if X[self.split_axis] < self.split_value:
            if self.left_subtree is not None:
                neighbors_distance_list = self.left_subtree.query_node(X, k, neighbors_distance_list)
        else:
            if self.right_subtree is not None:
                neighbors_distance_list = self.right_subtree.query_node(X, k, neighbors_distance_list)

        # Verify self
        neighbors_distance_list = sorted(neighbors_distance_list, key=lambda n: n[1], reverse=True)
        #print(X)
        #print(self.data[self.node_index])
        #print(self.kwargs)
        dist = self.distance_function(X, self.data[self.node_index], **self.kwargs)
        #print("verificou self")
        if len(neighbors_distance_list) > 0:
            if (dist < neighbors_distance_list[0][1]) or (len(neighbors_distance_list) < k):
                if len(neighbors_distance_list) >= k:
                    neighbors_distance_list[0] = (self.node_index, dist)
                else:
                    neighbors_distance_list.append((self.node_index, dist))
                #print("fez algo")
        else:
            neighbors_distance_list.append((self.node_index, dist))

        # Verify other subtree
        neighbors_distance_list = sorted(neighbors_distance_list, key=lambda n: n[1], reverse=True)
        aux = cp.deepcopy(self.kwargs)
        aux['index'] = self.split_axis
        dist = self.distance_function(X, self.data[self.node_index], **aux)
        del aux
        if (dist < neighbors_distance_list[0][1]) or (len(neighbors_distance_list) < k):
            #print("verificou outra")
            if X[self.split_axis] < self.split_value:
                if self.right_subtree is not None:
                    neighbors_distance_list = self.right_subtree.query_node(X, k, neighbors_distance_list)
                    #print("outra 1")
            else:
                if self.left_subtree is not None:
                    neighbors_distance_list = self.left_subtree.query_node(X, k, neighbors_distance_list)
                    #print("outra 2")
        #print(neighbors_distance_list)
        #print(str(len(neighbors_distance_list)))
        return neighbors_distance_list


    @property
    def _is_leaf(self):
        return ((self.left_subtree is None) and (self.right_subtree is None))

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