import numpy as np
from skmultiflow.data.base_stream import Stream
from sklearn.datasets import make_multilabel_classification
from skmultiflow.core.utils.validation import check_random_state


class MultilabelGenerator(Stream):
    """ MultilabelGenerator

    This generator creates a stream of samples for a multilabel problem. 
    It uses the make_multilabel_classification function from scikit-learn, 
    which creates a batch setting multilabel classification problem. These 
    samples are then sequentially fed by the next_sample function.

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
        If None, the random number generator is the RandomState instance used by `np.random`.

    Notes
    -----
    This is a wrapper for scikit-lean's `make_multilabel_classification`
        
    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.data.generators.multilabel_generator import MultilabelGenerator
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
        self.n_num_features = n_features
        self.random_state = check_random_state(random_state)
        self.__configure()

    def __configure(self):
        """ __configure

        Uses the make_multilabel_classification function from scikit-learn 
        to generate a multilabel classification problem. This problem will 
        be kept in memory and provided as demanded.


        """
        self.X, self.y = make_multilabel_classification(n_samples=self.n_samples,
                                                        n_features=self.n_features,
                                                        n_classes=self.n_targets,
                                                        n_labels=self.n_labels,
                                                        random_state=self.random_state)
        self.outputs_labels = ["target_" + str(i) for i in range(self.n_targets)]
        self.features_labels = ["att_num_" + str(i) for i in range(self.n_num_features)]

    def n_remaining_samples(self):
        return self.n_samples - self.sample_idx

    def has_more_samples(self):
        return self.n_samples - self.sample_idx > 0

    def next_sample(self, batch_size=1):
        """ next_sample
        
        Return batch_size samples from the X and y matrices stored in
        memory.
        
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

    def is_restartable(self):
        return True

    def restart(self):
        self.sample_idx = 0
        self.current_sample_x = None
        self.current_sample_y = None

    def get_n_cat_features(self):
        return self.n_cat_features

    def get_n_num_features(self):
        return self.n_num_features

    def get_n_features(self):
        return self.n_features

    def get_n_targets(self):
        return self.n_targets

    def get_feature_names(self):
        return self.features_labels

    def get_target_names(self):
        return self.outputs_labels

    def last_sample(self):
        return self.current_sample_x, self.current_sample_y

    def prepare_for_use(self):
        pass

    def get_name(self):
        return 'Multilabel Generator - {} targets'.format(self.n_targets)

    def get_targets(self):
        if self.n_targets == 1:
            return np.unique(self.y).tolist()
        else:
            return [np.unique(self.y[:, i]).tolist() for i in range(self.n_targets)]

    def get_class_type(self):
        return 'stream'

    def get_info(self):
        return 'MultilabelGenerator: n_samples: ' + str(self.n_samples) + \
               ' - n_features: ' + str(self.n_features) + \
               ' - n_targets: ' + str(self.n_targets) + \
               ' - n_labels:' + str(self.n_labels)
