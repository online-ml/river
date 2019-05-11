import numpy as np
from skmultiflow.data.base_stream import Stream
from sklearn.datasets import make_multilabel_classification
from skmultiflow.utils import check_random_state


class MultilabelGenerator(Stream):
    """ Creates a multi-label stream.

    This generator creates a stream of samples for a multi-label problem.
    It uses the make_multi-label_classification function from scikit-learn,
    which creates a batch setting multi-label classification problem. These
    samples are then sequentially yield by the next_sample method.

    Parameters
    ----------
    n_samples: int (Default: 40000)
        Total amount of samples to generate.

    n_features: int (Default: 100)
        Number of features to generate.

    n_targets: int (Default: 1)
        Number of targets to generate.
        
    n_labels: int (Default: 2)
        Average number of labels per instance.

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Notes
    -----
    This is a wrapper for scikit-lean's `make_multilabel_classification`
        
    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.data.multilabel_generator import MultilabelGenerator
    >>> # Setting up the stream
    >>> stream = MultilabelGenerator(n_samples=100, n_features=20, n_targets=4, n_labels=4, random_state=0)
    >>> stream.prepare_for_use()
    >>> # Retrieving one sample
    >>> stream.next_sample()
    (array([[3., 0., 1., 3., 6., 2., 5., 0., 5., 6., 3., 5., 1., 2., 0., 3.,
         3., 2., 2., 1.]]), array([[0, 1, 1, 1]]))
    >>> # Retrieving 10 samples
    >>> stream.next_sample(10)
    (array([[4., 0., 2., 6., 2., 2., 1., 1., 3., 1., 3., 0., 1., 4., 0., 1.,
         2., 2., 1., 1.],
        [2., 2., 1., 6., 4., 0., 3., 1., 2., 4., 2., 2., 1., 2., 2., 1.,
         3., 2., 1., 1.],
        [7., 3., 3., 5., 6., 1., 4., 3., 3., 1., 1., 1., 1., 1., 1., 1.,
         3., 2., 1., 8.],
        [1., 5., 1., 3., 4., 2., 2., 0., 4., 3., 2., 2., 2., 2., 3., 1.,
         5., 0., 2., 0.],
        [7., 3., 2., 7., 4., 6., 2., 1., 4., 1., 1., 0., 1., 0., 1., 0.,
         1., 1., 1., 4.],
        [0., 2., 1., 1., 6., 3., 4., 2., 5., 3., 0., 3., 0., 1., 3., 0.,
         3., 3., 2., 3.],
        [5., 1., 2., 3., 4., 1., 0., 3., 3., 3., 8., 0., 0., 2., 0., 0.,
         0., 2., 1., 1.],
        [2., 5., 6., 0., 5., 2., 5., 2., 5., 4., 1., 1., 4., 1., 1., 0.,
         1., 8., 3., 4.],
        [2., 4., 6., 2., 3., 8., 2., 2., 3., 3., 5., 1., 0., 0., 1., 4.,
         0., 1., 0., 3.],
        [4., 2., 2., 2., 6., 5., 3., 3., 6., 1., 1., 0., 2., 2., 1., 2.,
         3., 5., 1., 5.]]), array([[1, 1, 1, 1],
        [0, 1, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 1],
        [0, 1, 0, 0],
        [1, 1, 1, 0],
        [0, 1, 0, 0],
        [1, 1, 1, 1]]))
    >>> stream.n_remaining_samples()
    89
    >>> stream.has_more_samples()
    True

    """

    def __init__(self, n_samples=40000, n_features=20, n_targets=5, n_labels=2, random_state=None):
        super().__init__()
        self.X = None
        self.y = None
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_targets = n_targets
        self.n_labels = n_labels
        self.n_classes = 2
        self.n_num_features = n_features
        self.random_state = random_state
        self._random_state = None   # This is the actual random_state object used internally
        self.name = "Multilabel Generator"

    def prepare_for_use(self):
        """ Prepare the stream for usage

        Uses the make_multilabel_classification function from scikit-learn 
        to generate a multilabel classification problem. This problem will 
        be kept in memory and provided as demanded.


        """
        self._random_state = check_random_state(self.random_state)
        self.X, self.y = make_multilabel_classification(n_samples=self.n_samples,
                                                        n_features=self.n_features,
                                                        n_classes=self.n_targets,
                                                        n_labels=self.n_labels,
                                                        random_state=self._random_state)
        self.target_names = ["target_" + str(i) for i in range(self.n_targets)]
        self.feature_names = ["att_num_" + str(i) for i in range(self.n_num_features)]
        self.target_values = np.unique(self.y).tolist() if self.n_targets == 1 else\
            [np.unique(self.y[:, i]).tolist() for i in range(self.n_targets)]

    def next_sample(self, batch_size=1):
        """ Return batch_size samples from the X and y matrices stored in memory.
        
        Parameters
        ----------
        batch_size: int
            The number of samples to return.
        
        Returns
        -------
        tuple or tuple list
            Return a tuple with the features matrix and the labels matrix for 
            the batch_size samples that were requested.
        
        """
        self.sample_idx += batch_size
        try:
            self.current_sample_x = self.X[self.sample_idx - batch_size:self.sample_idx, :]
            self.current_sample_y = self.y[self.sample_idx - batch_size:self.sample_idx, :]
            if self.n_targets < 2:
                self.current_sample_y = self.current_sample_y.flatten()

        except IndexError:
            self.current_sample_x = None
            self.current_sample_y = None

        return self.current_sample_x, self.current_sample_y

    def restart(self):
        """ Restarts the stream
        """
        # Note: No need to regenerate the data, just reset the idx
        self.sample_idx = 0
        self.current_sample_x = None
        self.current_sample_y = None

    def n_remaining_samples(self):
        """
        Returns
        -------
        int
            Number of remaining samples.
        """
        return self.n_samples - self.sample_idx
