import copy as cp
from sklearn import linear_model
from skmultiflow.core.base import StreamModel
from skmultiflow.evaluation.metrics.metrics import *


class MultiOutputLearner(StreamModel):
    """ MultiOutputLearner
    
    A Multi-Output Learner learns to predict multiple outputs for each
    instance. The outputs may either be discrete (i.e., classification),
    or continuous (i.e., regression). This class takes any base learner
    (which by default is LogisticRegression) and builds a separate model
    for each output, and will distribute each instance to each model
    for individual learning and classification. 
    
    Use this meta learner to make single output predictors capable of learning 
    a multi output problem, by applying them individually to each output. In 
    the classification context, this is the "binary relevance" classifier.

    Parameters
    ----------
    h: classifier (extension of the BaseClassifier)
        This is the ensemble classifier type, each ensemble classifier is going 
        to be a copy of the h classifier.
        
    Examples
    --------
    # Imports
    >>> from skmultiflow.classification.multi_output_learner import MultiOutputLearner
    >>> from skmultiflow.core.pipeline import Pipeline
    >>> from skmultiflow.data.file_stream import FileStream
    >>> from sklearn.linear_model.perceptron import Perceptron
    >>> # Setup the file stream
    >>> stream = FileStream("skmultiflow/datasets/music.csv", 0, 6)
    >>> stream.prepare_for_use()
    >>> # Setup the MultiOutputLearner using sklearn Perceptrons
    >>> classifier = MultiOutputLearner(h=Perceptron())
    >>> # Setup the pipeline
    >>> pipe = Pipeline([('classifier', classifier)])
    >>> # Pre training the classifier with 150 samples
    >>> X, y = stream.next_sample(150)
    >>> pipe = pipe.partial_fit(X, y, target_values=stream.get_targets())
    >>> # Keeping track of sample count, true labels and predictions to later 
    >>> # compute the classifier's hamming score
    >>> count = 0
    >>> true_labels = []
    >>> predicts = []
    >>> while stream.has_more_samples():
    ...     X, y = stream.next_sample()
    ...     p = pipe.predict(X)
    ...     pipe = pipe.partial_fit(X, y)
    ...     predicts.extend(p)
    ...     true_labels.extend(y)
    ...     count += 1
    >>>
    >>> perf = hamming_score(true_labels, predicts)
    >>> print('Total samples analyzed: ' + str(count))
    >>> print("The classifier's static Hamming score    : " + str(perf))
    
    """

    h = None
    L = -1

    def __init__(self, h=linear_model.SGDClassifier(max_iter=100)):
        super().__init__()
        self.hop = h
        self.h = None
        self.L = None

    def __configure(self):
        self.h = [cp.deepcopy(self.hop) for j in range(self.L)]

    def fit(self, X, y, classes=None, weight=None):
        """ fit

        Fit the N classifiers, one for each classification task.

        Parameters
        ----------
        X : Numpy.ndarray of shape (n_samples, n_features)
            The array of samples used to fit the model.

        y: Array-like
            An array-like with the labels of all samples in X.

        classes: Not used.

        weight: Not used.

        Returns
        -------
        MultiOutputLearner
            self

        """
        N,L = y.shape
        self.L = L
        self.h = [cp.deepcopy(self.hop) for j in range(self.L)]

        for j in range(self.L):
            self.h[j].fit(X, y[:, j])
        return self

    def partial_fit(self, X, y, classes=None, weight=None):
        """ partial_fit

        Partially fit each of the classifiers on the X matrix and the 
        corresponding y matrix.

        Parameters
        ----------
        X : Numpy.ndarray of shape (n_samples, n_features)
            The array of samples used to fit the model.

        y: Numpy.ndarray of shape (n_samples, n_labels)
            An array-like with the labels of all samples in X.

        classes: Array-like
            Contains all labels that may appear in samples. It's an optional 
            parameter, except during the first partial_fit call, when it's 
            obligatory.

        weight: Array-like.
            Instance weight.

        Returns
        -------
        MultiOutputLearner
            self

        """
        Y = y

        N,self.L = Y.shape

        if self.h is None:
            self.__configure()

        for j in range(self.L):
            if "weight" in self.h[j].partial_fit.__code__.co_varnames:
                self.h[j].partial_fit(X, Y[:, j], classes, weight)
            elif "sample_weight" in self.h[j].partial_fit.__code__.co_varnames:
                self.h[j].partial_fit(X, Y[:, j], classes, weight)
            else:
                self.h[j].partial_fit(X, Y[:, j])

        return self

    def predict(self, X):
        ''' predict
            
        Iterates over all the classifiers, predicting with each one, to obtain 
        the multi output prediction.
        
        Parameters
        ----------
        X : Numpy.ndarray of shape (n_samples, n_features)
            The matrix of samples one wants to predict.
            
        Returns
        -------
        numpy.ndarray
            Numpy.ndarray of shape (n_samples, n_labels)
            All the predictions for the samples in X.
        '''
        N,D = X.shape
        Y = np.zeros((N,self.L))
        for j in range(self.L):
            Y[:,j] = self.h[j].predict(X)
        return Y

    def predict_proba(self, X):
        """ predict_proba
        
        Estimates the probability of each sample in X belonging to each of 
        the existing labels for each of the classification tasks.
        
        It's a simple call to all of the classifier's predict_proba function, 
        return the probabilities for all the classification problems.

        Parameters
        ----------
        X : Numpy.ndarray of shape (n_samples, n_features)
            The matrix of samples one wants to predict.

        Returns
        -------
        numpy.ndarray
            An array of shape (n_samples, n_classification_tasks, n_labels), in which 
            we store the probability that each sample in X belongs to each of the labels, 
            in each of the classification tasks.
        
        """
        N,D = X.shape
        P = zeros((N,self.L))
        for j in range(self.L):
            P[:,j] = self.h[j].predict_proba(X)[:,1]
        return P

    def get_info(self):
        return 'MultiOutputLearner: h: ' + str(self.h) + \
               ' - n_learners: ' + str(len(self.h)) + \
               ' - n_classification_tasks: ' + str(self.L)


    def score(self, X, y):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

def demo():
    import sys
    sys.path.append( '../data' )
    from skmultiflow.data.synth import make_logical

    X,Y = make_logical()
    N,L = Y.shape

    h = MultiOutputLearner(linear_model.SGDClassifier(max_iter=1000))
    h.fit(X, Y)

    p = h.predict(X)
    ham = hamming_score(Y, p)
    print(ham)

    # Test
    print(h.predict(X))
    print("vs")
    print(Y)

if __name__ == '__main__':
    demo()

