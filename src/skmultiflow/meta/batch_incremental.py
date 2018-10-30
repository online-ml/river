import copy as cp
import numpy as np
from skmultiflow.core.base import StreamModel
from sklearn.tree import DecisionTreeClassifier


class BatchIncremental(StreamModel):
    """ Batch Incremental.

    This is a wrapper that allows the application of any batch model to a 
    stream by incrementally building an ensemble of them. A window of examples 
    is collected, then used to train a new model, which is added to the 
    ensemble. A maximum number of models ensures memory use is finite (the 
    oldest model is deleted when this number is exceeded). 

    Parameters
    ----------
    base_estimator: StreamModel or sklearn model
        This is the ensemble learner type, each ensemble model is a copy of 
        this one.
        
    window_size (int)
        The size of the training window (batch), in other words, how many instances are kept for training.

    n_estimators (int)
        Number of estimators in the ensemble.

    Notes
    -----
    Not yet multi-label capable. 

    """

    def __init__(self, base_estimator=DecisionTreeClassifier(), window_size=100, n_estimators=100):
        self.window_size = window_size
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        # The ensemble 
        self.ensemble = []
        self.i = -1
        self.X_batch = None
        self.y_batch = None

    def fit(self, X, y, classes=None, weight=None):
        raise NotImplementedError

    def partial_fit(self, X, y=None, classes=None, weight=None):
        N, D = X.shape

        if self.i < 0:
            # No models yet -- initialize
            self.X_batch = np.zeros((self.window_size, D))
            self.y_batch = np.zeros(self.window_size)
            self.i = 0

        for n in range(N):
            # For each instance ...
            # TODO not very pythonic at the moment
            self.X_batch[self.i] = X[n]
            self.y_batch[self.i] = y[n]
            self.i = self.i + 1
            if self.i == self.window_size:
                # Get rid of the oldest model
                if len(self.ensemble) >= self.n_estimators:
                    self.ensemble.pop(0)
                # A new model
                h = cp.deepcopy(self.base_estimator)
                # Train it 
                h.fit(X=self.X_batch, y=self.y_batch.astype(int))
                # Add it
                self.ensemble.append(h)
                # Reset the window
                self.i = 0

        return self

    def predict_proba(self, X): 
        N, D = X.shape
        votes = np.zeros(N)
        if len(self.ensemble) <= 0:
            # No models yet, just predict zeros
            return votes
        for h_i in self.ensemble:
            # Add vote (normalized by number of models)
            votes = votes + 1. / len(self.ensemble) * h_i.predict(X)
        return votes

    def predict(self, X):
        votes = self.predict_proba(X)
        # Suppose a threshold of 0.5
        return (votes >= 0.5) * 1.

    def score(self, X, y):
        raise NotImplementedError

    def reset(self):
        self.ensemble = []
        self.i = -1
        self.X_batch = None
        self.y_batch = None

    def get_info(self):
        return 'BatchIncremental Classifier:' \
               ' - base_estimator: {}'.format(type(self.base_estimator).__name__) + \
               ' - window_size: {}'.format(self.window_size) + \
               ' - n_estimators: {}'.format(self.n_estimators)
