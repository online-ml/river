import sys
from skmultiflow.lazy import KNN
from skmultiflow.drift_detection import ADWIN
from skmultiflow.utils.data_structures import InstanceWindow
from skmultiflow.utils.utils import *


class KNNAdwin(KNN):
    """ K-Nearest Neighbors Classifier with ADWIN Change detector 
    
    This Classifier is an improvement from the regular KNN classifier, 
    as it is resistant to concept drift. It utilises the ADWIN change 
    detector to decide which samples to keep and which ones to forget, 
    and by doing so it regulates the sample window size.
     
    To know more about the ADWIN change detector, please visit 
    skmultiflow.classification.core.drift_detection.adwin

    It uses the regular KNN Classifier as a base class, with the 
    major difference that this class keeps a variable size window, 
    instead of a fixed size one and also it updates the adwin algorithm 
    at each partial_fit call.
    
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
        
    categorical_list: An array-like
        Each entry is the index of a categorical feature. May be requested 
        further filtering.
        
    Raises
    ------
    NotImplementedError: A few of the functions described here are not 
    implemented since they have no application in this context.
    
    ValueError: A ValueError is raised if the predict function is called 
    before at least k samples have been analyzed by the algorithm.
    
    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.lazy.knn_adwin import KNNAdwin
    >>> from skmultiflow.data.file_stream import FileStream
    >>> # Setting up the stream
    >>> stream = FileStream('skmultiflow/data/datasets/covtype.csv')
    >>> stream.prepare_for_use()
    >>> # Setting up the KNNAdwin classifier
    >>> knn_adwin = KNNAdwin(n_neighbors=8, leaf_size=40, max_window_size=2000)
    >>> # Pre training the classifier with 200 samples
    >>> X, y = stream.next_sample(200)
    >>> knn_adwin = knn_adwin.partial_fit(X, y)
    >>> # Keeping track of sample count and correct prediction count
    >>> n_samples = 0
    >>> corrects = 0
    >>> while n_samples < 5000:
    ...     X, y = stream.next_sample()
    ...     pred = knn_adwin.predict(X)
    ...     if y[0] == pred[0]:
    ...         corrects += 1
    ...     knn_adwin = knn_adwin.partial_fit(X, y)
    ...     n_samples += 1
    >>>
    >>> # Displaying the results
    >>> print('KNN usage example')
    >>> print(str(n_samples) + ' samples analyzed.')
    5000 samples analyzed.
    >>> print("KNNAdwin's performance: " + str(corrects/n_samples))
    KNNAdwin's performance: 0.7798

    """

    def __init__(self, n_neighbors=5, max_window_size=sys.maxsize, leaf_size=30, categorical_list=None):
        super().__init__(n_neighbors=n_neighbors, max_window_size=max_window_size, leaf_size=leaf_size,
                         categorical_list=categorical_list)
        self.adwin = ADWIN()
        self.window = None

    def reset(self):
        """ reset
        
        Resets the adwin algorithm as well as the base model 
        kept by the KNN base class.
        
        Returns
        -------
        KNNAdwin
            self
        
        """
        self.adwin = ADWIN()
        return super().reset()

    def fit(self, X, y, classes=None, weights=None):
        self.partial_fit(X, y, classes, weights)
        return self

    def partial_fit(self, X, y, classes=None, weight=None):
        """ partial_fit
        
        Partially fits the model. This is done by updating the window 
        with new samples while also updating the adwin algorithm. Then 
        we verify if a change was detected, and if so, the window is 
        correctly split at the drift moment.
        
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
        KNNAdwin
            self
        
        """
        r, c = get_dimensions(X)
        if self.window is None:
            self.window = InstanceWindow(max_size=self.max_window_size)

        for i in range(r):
            self.window.add_element(np.asarray([X[i]]), np.asarray([[y[i]]]))
            if self.window.n_samples >= self.n_neighbors:
                add = 1 if self.predict(np.asarray([X[i]])) == y[i] else 0
                self.adwin.add_element(add)
            else:
                self.adwin.add_element(0)

        if self.window.n_samples >= self.n_neighbors:
            changed = self.adwin.detected_change()

            if changed:
                if self.adwin.width < self.window.n_samples:
                    for i in range(self.window.n_samples, self.adwin.width, -1):
                        self.window.delete_element()
        return self

    def get_info(self):
        info = '{}:'.format(type(self).__name__)
        info += ' - n_neighbors: {}'.format(self.n_neighbors)
        info += ' - max_window_size: {}'.format(self.max_window_size)
        info += ' - leaf_size: {}'.format(self.leaf_size)
        return info
