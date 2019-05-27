import copy as cp

from skmultiflow.core import BaseSKMObject
from skmultiflow.lazy.distances import mixed_distance, euclidean_distance
from skmultiflow.utils.utils import *


class KDTree(BaseSKMObject):
    """ A K-dimensional tree implementation, adapted for k dimensional problems.
    
    Parameters
    ----------
    X: numpy.ndarray of shape (n_samples, n_features)
        The data upon which the object is going to create its tree.
    
    metric: a string representation of a metric.
        In the future custom metrics are going to be accepted.
        
    categorical_list: a list of the categorical feature's indexes.
        It may be used by some custom metric to differentiate between 
        categorical and numerical attributes.
        
    return_distance: bool
        Whether to return only the indexes of the closes neighbors or 
        also their distances.
        
    leaf_size: int
        The number of samples that can be held by a leaf node, at which 
        point the query algorithm will switch to brute force. The greater 
        the number the faster the tree will be build, and the slower the 
        query will be.
    
    kwargs: Additional keyword arguments
        Additional keyword arguments will be passed to the metric function.
    
    Raises
    ------
    May raise ValueErrors if the parameters passed don't have the correct types.
    
    Notes
    -----
    A KD Tree is a space partitioning (in axis-aligned hyper-rectangles) tree
    structure, where each node is associated with a feature and a splitting
    value. Each node can have up to two children nodes, with the property that
    all the samples stored by the left child have their feature value smaller
    then the node's splitting value, and all the samples stored by the right
    child have their relevant feature greater than or equal to the node's
    splitting value.

    The algorithm used to select the splitting feature and the splitting
    value for each node is called the sliding midpoint rule, as defined in
    Maneewongvatana and Mount 1999. The idea is that we don't need a balanced
    distribution of samples in all nodes, for the query time to be efficient,
    as long as there are 'fat' nodes around. This algorithm guarantees that
    the nodes won't become long and thin.

    This implementation is not faster than scikit-learn's implementation, 
    nor than scipy's implementation, but it allow users to use a custom 
    metric for the distance calculation.

    """
    _estimator_type = "data_structure"
    METRICS = ['mixed', 'euclidean']

    def __init__(self, X, metric='mixed', categorical_list=None, return_distance=False, leaf_size=40, **kwargs):
        super().__init__()

        self.distance_function = None

        self.metric = None

        # Accepts custom metrics
        if not callable(metric):
            if metric not in self.METRICS:
                raise ValueError("The metric '" + metric + "' is not supported by the KDTree.")
            else:
                if metric == self.METRICS[0]:
                    self.distance_function = mixed_distance
                    self.metric = self.METRICS[0]
                if metric == self.METRICS[1]:
                    self.distance_function = euclidean_distance
                    self.metric = self.METRICS[1]
        else:
            self.distance_function = metric
            self.metric = 'custom'

        self.X = np.asarray(X)
        if self.X.ndim != 2:
            raise ValueError("X should be a matrix, or array-like, of shape (n_samples, n_features).")

        self.n_samples, self.n_features = self.X.shape

        self.categorical_list = categorical_list
        self.return_distance = return_distance
        self.leaf_size = leaf_size

        self.kwargs = kwargs if kwargs is not None else {}

        distance_array = [0.0 for _ in range(self.n_features)]
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
        self.__build()

    def __build(self):
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

        Parameters
        ----------
        X: Array-like of size n_features or matrix of shape (n_samples, n_features)
            Stores the samples we want to find the nearest neighbors for.
        
        k: int
            The number of nearest neighbors to query for.
            
        Returns
        -------
        list or tuple list
            idx: List containing all the indexes from the closest neighbors (if return_distance is False)
            (dist, idx): Distance to neighbors and corresponding index ()if return_distance is True).
        
        """
        r, c = get_dimensions(X)
        dist_all, ind_all = [], []

        if (r == 1) and not isinstance(X[0], type([]) and not isinstance(X[0], type(np.ndarray([0])))):
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
        """ Radius based query
        
        Queries the tree based on a limited radius rather than a number of
        neighbors. 
        
        Parameters
        ----------
        X: Array-like of size n_features or matrix of shape (n_samples, n_features)
            Stores the samples we want to find the nearest neighbors for.
            
        r: float
            The maximum radius for the query
            
        Returns
        -------
        list or tuple list
            A list of indexes, or indexes and distances (if return_distance is True), 
            from all the neighbors within the maximum query radius.
        
        """
        raise NotImplementedError

    @property
    def _root(self):
        return self.root

    def get_info(self):
        info = '{}('.format(type(self).__name__)
        info += 'categorical_list={}, '.format(self.categorical_list)
        info += 'leaf_size={}, '.format(self.leaf_size)
        info += 'metric={}, '.format(self.metric)
        info += 'return_distance={})'.format(self.return_distance)
        return info


class KDTreeNode(object):
    """ A node from a KD Tree. A node object will store the indexes of its children's
    samples, and only a reference to the tree's complete data.
    
    Parameters
    ----------
    data: Numpy.ndarray of shape (n_samples, n_features)
        A reference to all the samples upon which the tree will be built.
        
    left_indexes: An array-like
        All the indexes from the samples that should be kept under the node's left 
        child.
        
    right_indexes: An array-like
        All the indexes from the samples that should be kept under the node's right 
        child.
        
    split_axis: int
        The node's chosen feature's index. This will be the node's splitting axis.
        
    split_value: int, float (numeric value)
        The node's splitting value.
        
    distance_function: A distance function.
        Any function that computes the distance between two samples.
        
    leaf_size: int
        The number of samples that can be stored in one leaf node, from which point 
        the algorithm switches to a brute-force approach.
        
    Notes
    -----
    The sliding midpoint rule, described in Maneewongvatana and Mount 1999, is the 
    algorithm of choice for building the KDTree. All the calculations are done by 
    the node's parent. This changes for the root node, in which case it's the 
    KDTree __build function that does all the calculations.
    
    """

    def __init__(self, data, left_indexes, right_indexes, split_axis, split_value, distance_function, leaf_size,
                 **kwargs):
        super().__init__()

        self.data = data

        self.left_subtree = None
        self.right_subtree = None
        self.left_indexes = left_indexes
        self.right_indexes = right_indexes

        # Here we assume left and right have the same dimensions.
        self.split_axis = split_axis
        self.split_value = split_value
        self.leaf_size = leaf_size
        self.is_leaf = False
        self.leaf_indexes = None

        self.distance_function = distance_function
        self.kwargs = kwargs

        self.__start_node()

    def __start_node(self):
        """ Start a node
        
        This functions will start up a node, based on the sliding midpoint rule.
        
        Returns
        -------
        KDTree
            self
        
        """

        # Checking if this could be a leaf node
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
            # Handling left subtree
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

                    # Get the split point
                    split = (maxval + minval) / 2

                    # Split the indexes between left and right child
                    left = np.nonzero(data < split)[0]
                    left = np.asarray([self.left_indexes[k] for k in left])
                    right = np.nonzero(data >= split)[0]
                    right = np.asarray([self.left_indexes[k] for k in right])

                    # If there's a child with no indexes, while the other has more than
                    # leaf_size indexes, we slide the cutting point towards the 'fat'
                    # size until there's at least one index on each child.
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

                    # Creates the left subtree
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

                    # Get the split point
                    split = (maxval + minval) / 2

                    # Split the indexes between left and right child
                    left = np.nonzero(data < split)[0]
                    left = np.asarray([self.right_indexes[k] for k in left])
                    right = np.nonzero(data >= split)[0]
                    right = np.asarray([self.right_indexes[k] for k in right])

                    # If there's a child with no indexes, while the other has more than
                    # leaf_size indexes, we slide the cutting point towards the 'fat'
                    # size until there's at least one index on each child.
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

                    # Creates the right subtree
                    self.right_subtree = KDTreeNode(data=self.data, left_indexes=left, right_indexes=right, split_axis=d,
                                                    split_value=split, distance_function=self.distance_function,
                                                    leaf_size=self.leaf_size, **self.kwargs)

                else:
                    self.right_subtree = None
            else:
                self.right_subtree = None

        return self

    def query_node(self, X, k, neighbors_distance_list):
        """ Query a node
         
        Queries a node and all of it's sub nodes, if there is a chance of finding 
        a nearest neighbor in that branch. 

        Parameters
        ----------
        X: Array-like
            The sample (only one) wants to find the nearest neighbors for.
            
        k: int
            The number of nearest neighbors to query for.
        
        neighbors_distance_list: A list of tuples
            A list containing tuples that represent the current nearest 
            neighbors found. The tuples are stored in the format (index, distance)
            
        Returns
        -------
        list
            An updated version of the neighbors_distance_list
         
        """
        # In case there is no more subtrees
        if not self:
            return neighbors_distance_list

        # If the node is a leaf, adopt the brute-force strategy
        if self.is_leaf:
            for i in range(len(self.leaf_indexes)):
                dist = self.distance_function(instance_one=self.data[self.leaf_indexes[i]],
                                              instance_two=X, **self.kwargs)
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

        # If the node is an inner node the regular query strategy is used
        else:
            # Advance in the tree structure until a leaf node is reached
            if X[self.split_axis] < self.split_value:
                if self.left_subtree is not None:
                    neighbors_distance_list = self.left_subtree.query_node(X, k, neighbors_distance_list)
            else:
                if self.right_subtree is not None:
                    neighbors_distance_list = self.right_subtree.query_node(X, k, neighbors_distance_list)

            # Check the other branch if it's possible that there is a nearest
            # neighbor in that branch
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
