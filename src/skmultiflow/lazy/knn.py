from skmultiflow.core.base import StreamModel
from skmultiflow.utils.data_structures import InstanceWindow
import sklearn.neighbors as sk
from skmultiflow.utils.utils import *


class KNN(StreamModel):
    """ K-Nearest Neighbors Classifier
    
    This is a non-parametric classification method. The output of this
    algorithm are the n_neighbors closest training examples to the query sample
    X.
    
    It works by keeping track of a fixed number of training samples, in 
    our case it keeps track of the last max_window_size training samples.
    Then, whenever a query request is executed, the algorithm will search 
    its stored samples and find the closest ones using a selected distance 
    metric.
    
    To store the samples, while reducing search times, we use a structure 
    called KD Tree (a K Dimensional Tree, for n_neighbors dimensional problems).
    Although we do have our own KDTree implementation, which accepts 
    custom metrics, we recommend using the standard scikit-learn KDTree,  
    that even though doesn't accept custom metrics, is optimized and will 
    function faster.
    
    Parameters
    ----------
    n_neighbors: int
        The number of nearest neighbors to search for.
        
    max_window_size: int
        The maximum size of the window storing the last viewed samples.
        
    leaf_size: int
        The maximum number of samples that can be stored in one leaf node, 
        which determines from which point the algorithm will switch for a 
        brute-force approach. The bigger this number the faster the tree 
        construction time, but the slower the query time will be.
        
    categorical_list: array-like
        Each entry is the index of a categorical feature. May be requested 
        further filtering.
    
    Raises
    ------
    NotImplementedError: A few of the functions described here are not 
    implemented since they have no application in this context.
    
    ValueError: A ValueError is raised if the predict function is called 
    before at least n_neighbors samples have been analyzed by the algorithm.
    
    Notes
    -----
    For a KDTree functionality explanation, please see our KDTree 
    documentation, under skmultiflow.lazy.neighbors.kdtree.
    
    This classifier is not optimal for a mixture of categorical and 
    numerical features.
    
    If you wish to use our KDTree implementation please refer to this class' 
    function __predict_proba
    
    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.lazy.knn import KNN
    >>> from skmultiflow.data.file_stream import FileStream
    >>> # Setting up the stream
    >>> stream = FileStream('skmultiflow/data/datasets/sea_big.csv', -1, 1)
    >>> stream.prepare_for_use()
    >>> # Pre training the classifier with 200 samples
    >>> X, y = stream.next_sample(200)
    >>> knn = KNN(n_neighbors=8, max_window_size=2000, leaf_size=40)
    >>> knn.partial_fit(X, y)
    >>> # Preparing the processing of 5000 samples and correct prediction count
    >>> n_samples = 0
    >>> corrects = 0
    >>> while n_samples < 5000:
    ...     X, y = stream.next_sample()
    ...     my_pred = knn.predict(X)
    ...     if y[0] == my_pred[0]:
    ...         corrects += 1
    ...     knn = knn.partial_fit(X, y)
    ...     n_samples += 1
    >>>
    >>> # Displaying results
    >>> print('KNN usage example')
    >>> print(str(n_samples) + ' samples analyzed.')
    5000 samples analyzed.
    >>> print("KNN's performance: " + str(corrects/n_samples))
    KNN's performance: 0.868
    
    """

    def __init__(self, n_neighbors=5, max_window_size=1000, leaf_size=30, categorical_list=None):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.max_window_size = max_window_size
        self.c = 0
        self.window = InstanceWindow(max_size=max_window_size, dtype=float)
        self.first_fit = True
        self.classes = []
        self.leaf_size = leaf_size
        if categorical_list is None:
            self.categorical_list = []

    def fit(self, X, y, classes=None, weight=None):
        """ fit
        
        Fits the model on the samples X and targets y. This is actually the 
        function as the partial fit.
        
        For the K-Nearest Neighbors Classifier, fitting the model is the 
        equivalent of inserting the newer samples in the observed window, 
        and if the size_limit is reached, removing older results. To store 
        the viewed samples we use a InstanceWindow object. For this class' 
        documentation please visit skmultiflow.core.utils.data_structures
        
        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            The data upon which the algorithm will create its model.
            
        y: Array-like
            An array-like containing the classification targets for all 
            samples in X.
            
        classes: Not used.

        weight: Not used.
        
        Returns
        -------
        KNN
            self
        
        """
        r, c = get_dimensions(X)
        if classes is not None:
            self.classes = list(set().union(self.classes, classes))

        for i in range(r):
            self.window.add_element(np.asarray([X[i]]), np.asarray([[y[i]]]))
        return self

    def partial_fit(self, X, y, classes=None, weight=None):
        """ partial_fit
        
        Fits the model on the samples X and targets y.
        
        For the K-Nearest Neighbors Classifier, fitting the model is the 
        equivalent of inserting the newer samples in the observed window, 
        and if the size_limit is reached, removing older results. To store 
        the viewed samples we use a InstanceWindow object. For this class' 
        documentation please visit skmultiflow.core.utils.data_structures
        
        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            The data upon which the algorithm will create its model.
            
        y: Array-like
            An array-like containing the classification targets for all 
            samples in X.

        classes: Not used.

        weight: Not used.
        
        Returns
        -------
        KNN
            self
        
        """
        r, c = get_dimensions(X)

        for i in range(r):
            self.window.add_element(np.asarray([X[i]]), np.asarray([[y[i]]]))
        return self

    def reset(self):
        self.window.reset()
        return self

    def predict(self, X):
        """ predict
        
        Predicts the label of the X sample, by searching the KDTree for 
        the n_neighbors-Nearest Neighbors.
        
        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            All the samples we want to predict the label for.
            
        Returns
        -------
        list
            A list containing the predicted labels for all instances in X.
        
        """
        r, c = get_dimensions(X)
        proba = self.predict_proba(X)
        predictions = []
        for i in range(r):
            predictions.append(np.argmax(proba[i]))
        return np.array(predictions)

    def predict_proba(self, X):
        """ predict_proba
         
        Calculates the probability of each sample in X belonging to each 
        of the labels, based on the knn algorithm.
        
        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
        
        Raises
        ------
        ValueError: If there is an attempt to call this function before, 
        at least, n_neighbors samples have been analyzed by the learner, a ValueError
        is raised.
        
        Returns
        -------
        numpy.ndarray
            An array of shape (n_samples, n_features), in which each outer entry is 
            associated with the X entry of the same index. And where the list in 
            index [i] contains len(self.target_value) elements, each of which represents
            the probability that the i-th sample of X belongs to a certain label.
         
        """
        if self.window is None or self.window.n_samples < self.n_neighbors:
            raise ValueError("KNN must be partially fitted on n_neighbors samples before doing any prediction.")
        proba = []
        r, c = get_dimensions(X)

        self.classes = list(set().union(self.classes, np.unique(self.window.get_targets_matrix())))

        new_dist, new_ind = self.__predict_proba(X)
        for i in range(r):
            votes = [0.0 for _ in range(int(max(self.classes) + 1))]
            for index in new_ind[i]:
                votes[int(self.window.get_targets_matrix()[index])] += 1. / len(new_ind[i])
            proba.append(votes)

        return np.array(proba)

    def __predict_proba(self, X):
        """ __predict_proba
        
        Private implementation of the predict_proba method.
        
        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
        
        Returns
        -------
        tuple list
            One list with the k-nearest neighbor's distances and another 
            one with their indexes.
        
        Notes
        -----
        If you wish to use our own KDTree implementation please comment 
        the third line of this function and uncomment the first and 
        second lines.
        
        """
        # tree = KDTree(self.window.get_attributes_matrix(), metric='euclidean',
        #              categorical_list=self.categorical_list, return_distance=True)

        tree = sk.KDTree(self.window.get_attributes_matrix(), self.leaf_size, metric='euclidean')
        dist, ind = tree.query(np.asarray(X), k=self.n_neighbors)
        return dist, ind

    def score(self, X, y):
        raise NotImplementedError

    def get_info(self):
        info = '{}:'.format(type(self).__name__)
        info += ' - n_neighbors: {}'.format(self.n_neighbors)
        info += ' - max_window_size: {}'.format(self.max_window_size)
        info += ' - leaf_size: {}'.format(self.leaf_size)
        return info
