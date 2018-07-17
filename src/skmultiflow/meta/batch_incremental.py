import copy as cp
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
    h: learner (extension of the BaseClassifier)
        This is the ensemble learner type, each ensemble model is a copy of 
        this one.
        
    window_size: int
        The size of the ensemble, in other words, how many classifiers to train.
        
    ensemble_length: int
        The maximum size of the ensemble, i.e., the maximum number of
        classifiers to store at any point in time.

    Notes
    -----
    Not yet multi-label capable. 

    """

    def __init__(self, h=DecisionTreeClassifier, window_size=100, ensemble_length=100):
        self.window_size = window_size
        self.max_models = ensemble_length
        self.h = h
        # The ensemble 
        self.H = []
        self.i = -1
        self.X_batch = None
        self.y_batch = None

    def fit(self, X, y):
        raise NotImplementedError

    def partial_fit(self, X, y=None):
        N,D = X.shape

        if self.i < 0:
            # No models yet -- initialize
            self.X_batch = zeros((self.window_size,D))
            self.y_batch = zeros(self.window_size)
            self.i = 0

        for n in range(N):
            # For each instance ...
            # (TODO: not very python-esque ot the moment)
            self.X_batch[self.i] = X[n]
            self.y_batch[self.i] = y[n]
            self.i = self.i + 1
            if self.i == self.window_size:
                # Get rid of the oldest model
                if len(self.H) >= self.max_models:
                    self.H.pop(0)
                # A new model
                h = cp.deepcopy(h)
                # Train it 
                h.fit(self.X_batch,self.y_batch)
                # Add it
                self.H.append(h)
                # Reset the window
                self.i = 0

        return self

    def predict_proba(self, X): 
        N,D = X.shape
        votes = zeros(N)
        if len(self.H) <= 0:
            # No models yet, just predict zeros
            return votes
        for h_i in self.H:
            # Add vote (normalized by number of models)
            votes = votes + 1./len(self.H) * h_i.predict(X)
        return votes

    def predict(self, X):
        votes = self.predict_proba(X)
        # Suppose a threshold of 0.5
        return (votes >= 0.5) * 1.

