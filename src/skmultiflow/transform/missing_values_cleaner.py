import numpy as np

from scipy import stats

from skmultiflow.transform.base_transform import StreamTransform
from skmultiflow.utils import FastBuffer, get_dimensions


class MissingValuesCleaner(StreamTransform):
    """ Fills missing values with some defined value.

    Provides a simple way to replace missing values in data samples with some value. The imputation value
    can be set via a set of imputation strategies.
    
    Parameters
    ----------
    missing_value: int, float or list (Default: numpy.nan)
        Missing value to replace
    
    strategy: string (Default: 'zero')
        The strategy adopted to find the missing value replacement. It can 
        be one of the following: 'zero', 'mean', 'median', 'mode', 'custom'.
    
    window_size: int (Default: 200)
        Defines the window size for the 'mean', 'median' and 'mode' strategies.
    
    new_value: int (Default: 1)
        This is the replacement value in case the chosen strategy is 'custom'.
        
    Examples
    --------
    >>> # Imports
    >>> import numpy as np
    >>> from skmultiflow.data.file_stream import FileStream
    >>> from skmultiflow.transform.missing_values_cleaner import MissingValuesCleaner
    >>> # Setting up a stream
    >>> stream = FileStream('skmultiflow/data/datasets/covtype.csv', -1, 1)
    >>> stream.prepare_for_use()
    >>> # Setting up the filter to substitute values -47 by the median of the 
    >>> # last 10 samples
    >>> cleaner = MissingValuesCleaner(-47, 'median', 10)
    >>> X, y = stream.next_sample(10)
    >>> X[9, 0] = -47
    >>> # We will use this list to keep track of values
    >>> data = []
    >>> # Iterate over the first 9 samples, to build a sample window
    >>> for i in range(9):
    >>>     X_transf = cleaner.partial_fit_transform([X[i].tolist()])
    >>>     data.append(X_transf[0][0])
    >>>
    >>> # Transform last sample. The first feature should be replaced by the list's 
    >>> # median value
    >>> X_transf = cleaner.partial_fit_transform([X[9].tolist()])
    >>> np.median(data)

    Notes
    -----
    A missing value in a sample can be coded in many different ways, but the
    most common one is to use numpy's NaN, that's why that is the default
    missing value parameter.

    The user should choose the correct substitution strategy for his use
    case, as each strategy has its pros and cons. The strategy can be chosen
    from a set of predefined strategies, which are: 'zero', 'mean', 'median',
    'mode', 'custom'.

    Notice that `MissingValuesCleaner` can actually be used to replace arbitrary
    values.

    """

    def __init__(self, missing_value=np.nan, strategy='zero', window_size=200, new_value=1):
        super().__init__()
        if isinstance(missing_value, list):
            self.missing_value = missing_value
        else:
            self.missing_value = [missing_value]
        self.strategy = strategy
        self.window_size = window_size
        self.window = None
        self.new_value = new_value

        self.__configure()

    def __configure(self):
        if self.strategy in ['mean', 'median', 'mode']:
            self.window = FastBuffer(max_size=self.window_size)

    def transform(self, X):
        """ transform
        
        Does the transformation process in the samples in X.
        
        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The sample or set of samples that should be transformed.
        
        """
        r, c = get_dimensions(X)
        for i in range(r):
            if self.strategy in ['mean', 'median', 'mode']:
                self.window.add_element([X[i][:]])
            for j in range(c):
                if X[i][j] in self.missing_value or np.isnan(X[i][j]):
                    X[i][j] = self._get_substitute(j)

        return X

    def _get_substitute(self, column_index):
        """ _get_substitute
        
        Computes the replacement for a missing value.
        
        Parameters
        ----------
        column_index: int
            The index from the column where the missing value was found.
            
        Returns
        -------
        int or float
            The replacement.
        
        """
        if self.strategy == 'zero':
            return 0
        elif self.strategy == 'mean':
            if not self.window.is_empty():
                return np.nanmean(np.array(self.window.get_queue())[:, column_index])
            else:
                return self.new_value
        elif self.strategy == 'median':
            if not self.window.is_empty():
                return np.nanmedian(np.array(self.window.get_queue())[:, column_index])
            else:
                return self.new_value
        elif self.strategy == 'mode':
            if not self.window.is_empty():
                return stats.mode(np.array(self.window.get_queue())[:, column_index],
                                  nan_policy='omit')[0]
            else:
                return self.new_value
        elif self.strategy == 'custom':
            return self.new_value

    def partial_fit_transform(self, X, y=None):
        """ partial_fit_transform
        
        Partially fits the model and then apply the transform to the data.
        
        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The sample or set of samples that should be transformed.
            
        y: Array-like
            The true labels.
         
        Returns
        -------
        numpy.ndarray of shape (n_samples, n_features)
            The transformed data.
        
        """
        X = self.transform(X)

        return X

    def partial_fit(self, X, y=None):
        """ partial_fit
        
        Partial fits the model.
        
        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The sample or set of samples that should be transformed.
            
        y: Array-like
            The true labels.
        
        Returns
        -------
        MissingValuesCleaner
            self
        
        """
        X = np.asarray(X)
        if self.strategy in ['mean', 'median', 'mode']:
            self.window.add_element(X)
        return self
