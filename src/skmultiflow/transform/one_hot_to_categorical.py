import numpy as np

from skmultiflow.transform.base_transform import StreamTransform
from skmultiflow.utils.utils import get_dimensions


class OneHotToCategorical(StreamTransform):
    """ Transforms one-hot encoded data into categorical feature(s).
    
    Receives a features matrix, with some binary features (one-hot), and transform them into single categorical
    feature.
    
    Parameters
    ----------
    categorical_list: list of lists
        Each inner list contains all the attribute indexes that are associated with
        the same categorical feature.
    
    """

    def __init__(self, categorical_list):
        super().__init__()
        self.categorical_list = categorical_list

    def transform(self, X):
        """ transform
        
        Transform one hot features in the X matrix into int coded 
        categorical features.
        
        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The sample or set of samples that should be transformed.
         
        Returns
        -------
        numpy.ndarray
            The transformed data.
        
        """
        r, c = get_dimensions(X)

        new_width = c
        for i in range(len(self.categorical_list)):
            new_width -= len(self.categorical_list[i]) - 1

        ret = np.zeros((0, new_width), dtype=X.dtype)
        for i in range(r):
            ret = np.concatenate((ret, self._transform(X[i, :], new_width)), axis=0)

        return ret

    def _transform(self, X, new_size):
        ret = np.zeros((1,new_size), dtype=X.dtype)
        list_index = 0
        i = 0
        new_i = 0
        while i < X.size:
            if i in self.categorical_list[list_index]:
                c = 0
                found = False
                for j in range(len(self.categorical_list[list_index])):
                    if X[i+j] == 1:
                        found = True
                    if not found:
                        c += 1
                i += len(self.categorical_list[list_index])
                list_index += 1
                ret[0][new_i] = c
                new_i += 1
            else:
                ret[0][new_i] = X[i]
                new_i += 1
                i += 1

        return ret

    def fit(self, X, y):
        return self

    def partial_fit_transform(self, X, y=None, classes=None):
        return self.transform(X)

    def partial_fit(self, X, y=None, classes=None):
        return self
