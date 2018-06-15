from skmultiflow.core.base import StreamModel
from skmultiflow.core.utils.data_structures import InstanceWindow
import sklearn.neighbors as sk
from skmultiflow.core.utils.utils import *


class KNN(StreamModel):
    """ K-Nearest Neighbors Classifier
    
    This is a non-parametric classification method. The output of this
    algorithm are the k closest training examples to the query sample 
    X.
    
    It works by keeping track of a fixed number of training samples, in 
    our case it keeps track of the last max_window_size training samples.
    Then, whenever a query request is executed, the algorithm will search 
    its stored samples and find the closest ones using a selected distance 
    metric.
    
    To store the samples, while reducing search times, we use a structure 
    called KD Tree (a K Dimensional Tree, for k dimensional problems). 
    Although we do have our own KDTree implementation, which accepts 
    custom metrics, we recommend using the standard scikit-learn KDTree,  
    that even though doesn't accept custom metrics, is optimized and will 
    function faster.
    
    Parameters
    ----------
    k: int
        The number of nearest neighbors to search for.
        
    max_window_size: int
        The maximum size of the window storing the last viewed samples.
        
    leaf_size: int
        The maximum number of samples that can be stored in one leaf node, 
        which determines from which point the algorithm will switch for a 
        brute-force approach. The bigger this number the faster the tree 
        construction time, but the slower the query time will be.
        
    categorical_list: An array-like
        Each entry is the index of a categorical feature. May be requested 
        further filtering.
    
    Raises
    ------
    NotImplementedError: A few of the functions described here are not 
    implemented since they have no application in this context.
    
    ValueError: A ValueError is raised if the predict function is called 
    before at least k samples have been analyzed by the algorithm.
    
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
    >>> from skmultiflow.classification.lazy.knn import KNN
    >>> from skmultiflow.data.file_stream import FileStream
    >>> # Setting up the stream
    >>> stream = FileStream('skmultiflow/datasets/sea_big.csv', -1, 1)
    >>> stream.prepare_for_use()
    >>> # Pre training the classifier with 200 samples
    >>> X, y = stream.next_sample(200)
    >>> knn = KNN(k=8, max_window_size=2000, leaf_size=40)
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

    def __init__(self, k=5, max_window_size=1000, leaf_size=30, categorical_list=None):
        super().__init__()
        self.k = k
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
        if self.window is None:
            self.window = InstanceWindow(max_size=self.max_window_size)

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
        if self.window is None:
            self.window = InstanceWindow(max_size=self.max_window_size)

        for i in range(r):
            self.window.add_element(np.asarray([X[i]]), np.asarray([[y[i]]]))
        return self

    def reset(self):
        self.window = None
        return self

    def predict(self, X):
        """ predict
        
        Predicts the label of the X sample, by searching the KDTree for 
        the k-Nearest Neighbors.
        
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
        probs = self.predict_proba(X)
        preds = []
        for i in range(r):
            preds.append(self.classes[probs[i].index(np.max(probs[i]))])
        return preds

    def _predict(self, X):
        raise NotImplementedError

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
        at least, k samples have been analyzed by the learner, a ValueError 
        is raised.
        
        Returns
        -------
        numpy.ndarray
            An array of shape (n_samples, n_features), in which each outer entry is 
            associated with the X entry of the same index. And where the list in 
            index [i] contains len(self.target_values) elements, each of which represents
            the probability that the i-th sample of X belongs to a certain label.
         
        """
        if self.window is None:
            raise ValueError("KNN should be partially fitted on at least k samples before doing any prediction.")
        if self.window._num_samples < self.k:
            raise ValueError("KNN should be partially fitted on at least k samples before doing any prediction.")
        probs = []
        r, c = get_dimensions(X)

        self.classes = list(set().union(self.classes, np.unique(self.window.get_targets_matrix())))

        new_dist, new_ind = self.__predict_proba(X)

        for i in range(r):
            classes = [0 for j in range(len(self.classes))]
            for index in new_ind[i]:
                classes[self.classes.index(self.window.get_targets_matrix()[index])] += 1
            probs.append([x/len(new_ind) for x in classes])

        return probs

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
        #tree = KDTree(self.window.get_attributes_matrix(), metric='euclidean',
        #              categorical_list=self.categorical_list, return_distance=True)

        tree = sk.KDTree(self.window.get_attributes_matrix(), self.leaf_size, metric='euclidean')
        dist, ind = tree.query(np.asarray(X), k=self.k)
        return dist, ind

    def score(self, X, y):
        raise NotImplementedError

    def get_info(self):
        return 'KNN Classifier: max_window_size: ' + str(self.max_window_size) + \
            ' - leaf_size: ' + str(self.leaf_size)