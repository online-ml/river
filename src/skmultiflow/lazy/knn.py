from skmultiflow.core import BaseSKMObject, ClassifierMixin
from skmultiflow.utils.data_structures import InstanceWindow
import sklearn.neighbors as sk
from skmultiflow.utils.utils import *


class KNN(BaseSKMObject, ClassifierMixin):
    """ K-Nearest Neighbors classifier.
    
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
    n_neighbors: int (default=5)
        The number of nearest neighbors to search for.
        
    max_window_size: int (default=1000)
        The maximum size of the window storing the last viewed samples.
        
    leaf_size: int (default=30)
        The maximum number of samples that can be stored in one leaf node, 
        which determines from which point the algorithm will switch for a 
        brute-force approach. The bigger this number the faster the tree 
        construction time, but the slower the query time will be.
        
    nominal_attributes: numpy.ndarray (optional, default=None)
        List of Nominal attributes. If empty, then assume that all attributes are numerical.
    
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
    >>> from skmultiflow.lazy import KNN
    >>> from skmultiflow.data import SEAGenerator
    >>> # Setting up the stream
    >>> stream = SEAGenerator(random_state=1, noise_percentage=.1)
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
    >>> print('{} samples analyzed.'.format(n_samples))
    5000 samples analyzed.
    >>> print("KNN's performance: {}".format(corrects/n_samples))
    KNN's performance: 0.8788
    
    """

    def __init__(self, n_neighbors=5, max_window_size=1000, leaf_size=30, nominal_attributes=None):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.max_window_size = max_window_size
        self.c = 0
        self.window = InstanceWindow(max_size=max_window_size, dtype=float)
        self.first_fit = True
        self.classes = []
        self.leaf_size = leaf_size
        self.nominal_attributes = nominal_attributes
        if self.nominal_attributes is None:
            self._nominal_attributes = []

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """ Partially fits the model on the samples X and corresponding targets y.
        
        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            The data upon which the algorithm will create its model.
            
        y: Array-like
            An array-like containing the classification targets for all 
            samples in X.

        classes: numpy.ndarray, optional (default=None)
            Array with all possible/known classes.

        sample_weight: Not used.
        
        Returns
        -------
        KNN
            self

        Notes
        -----
        For the K-Nearest Neighbors Classifier, fitting the model is the
        equivalent of inserting the newer samples in the observed window,
        and if the size_limit is reached, removing older results. To store
        the viewed samples we use a InstanceWindow object. For this class'
        documentation please visit skmultiflow.core.utils.data_structures

        """
        r, c = get_dimensions(X)

        if classes is not None:
            self.classes = list(set().union(self.classes, classes))

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
            raise ValueError("KNN must be (partially) fitted on n_neighbors samples before doing any prediction.")
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
        
        """
        # To use our own KDTree implementation please replace it as follows
        # tree = KDTree(self.window.get_attributes_matrix(), metric='euclidean',
        #              nominal_attributes=self._nominal_attributes, return_distance=True)

        tree = sk.KDTree(self.window.get_attributes_matrix(), self.leaf_size, metric='euclidean')
        dist, ind = tree.query(np.asarray(X), k=self.n_neighbors)
        return dist, ind
