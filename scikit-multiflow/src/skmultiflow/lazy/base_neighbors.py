from skmultiflow.core import BaseSKMObject
from skmultiflow.utils import SlidingWindow

from sklearn.neighbors import KDTree


class BaseNeighbors(BaseSKMObject):
    """ Base class for neighbors-based estimators
    """
    def __init__(self,
                 n_neighbors=5,
                 max_window_size=1000,
                 leaf_size=30,
                 metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.max_window_size = max_window_size
        self.leaf_size = leaf_size
        if metric not in self.valid_metrics():
            raise ValueError("Invalid metric: {}.\n"
                             "Valid options are: {}".format(metric,
                                                            self.valid_metrics()))
        self.metric = metric
        self.data_window = SlidingWindow(window_size=max_window_size)

    def _get_neighbors(self, X):
        tree = KDTree(self.data_window.features_buffer, self.leaf_size, metric=self.metric)
        dist, idx = tree.query(X=X,
                               k=self.n_neighbors)
        return dist, idx

    def reset(self):
        """ Reset estimator.
        """
        self.data_window.reset()
        return self

    @staticmethod
    def valid_metrics():
        """ Get valid distance metrics for the KDTree. """
        return KDTree.valid_metrics

