import numpy as np
from skmultiflow.data.base_instance_stream import BaseInstanceStream
from sklearn.datasets import make_multilabel_classification
from skmultiflow.core.utils.validation import check_random_state


class MultilabelGenerator(BaseInstanceStream):
    """ MultilabelGenerator

    This generator creates a stream of samples for a multilabel problem. 
    It uses the make_multilabel_classification function from scikit-learn, 
    which creates a batch setting multilabel classification problem. These 
    samples are then sequentially fed by the next_instance function.

    Parameters
    ----------
    n_samples: int (Default: 40000)
        Total amount of samples to generate.

    n_features: int (Default: 100)
        Number of features to generate.

    n_targets: int (Default: 1)
        Number of targeting tasks to generate.
        
    n_labels: int (Default: 2)
        Number of labels to generate.

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by `np.random`.
        
    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.data.generators.multilabel_generator import MultilabelGenerator
    >>> # Setting up the stream
    >>> stream = MultilabelGenerator(n_samples=100, n_features=20, n_targets=4, n_labels=4, random_state=0)
    >>> stream.prepare_for_use()
    >>> # Retrieving one sample
    >>> stream.next_instance()
    (array([[3., 0., 1., 3., 6., 2., 5., 0., 5., 6., 3., 5., 1., 2., 0., 3.,
         3., 2., 2., 1.]]), array([[0, 1, 1, 1]]))
    >>> # Retrieving 10 samples
    >>> stream.next_instance(10)
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
    >>> stream.estimated_remaining_instances()
    89
    >>> stream.has_more_instances()
    True

    """

    def __init__(self, n_samples=40000, n_features=20, n_targets=5, n_labels=2, random_state=None):
        super().__init__()
        self.X = None
        self.y = None
        self.n_samples = 0
        self.n_features = 0
        self.n_classes = 0
        self.n_outputs = 0
        self.num_numerical_attributes = 0
        self.num_nominal_attributes = 0
        self.num_values_per_nominal_att = 0
        self.instance_index = 0
        self.current_instance_y = None
        self.current_instance_x = None
        self.random_state = check_random_state(random_state)
        self.__configure(n_samples, n_features, n_targets, n_labels)

    def __configure(self, n_samples, n_features, n_targets, n_outputs):
        """ __configure

        Uses the make_multilabel_classification function from scikit-learn 
        to generate a multilabel classification problem. This problem will 
        be kept in memory and provided as demanded.

        Parameters
        ----------
        n_samples: int
            Total amount of samples to generate.

        n_features: int
            Number of features to generate.

        n_targets: int
            Number of targeting tasks to generate.
        
        n_outputs: int
            Number of outputs to generate.

        """
        self.X, self.y = make_multilabel_classification(n_samples=n_samples, n_features=n_features, n_classes=n_targets,
                                                        n_labels=n_outputs, random_state=self.random_state)
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_targets
        self.n_outputs = n_outputs
        self.num_numerical_attributes = n_features
        self.class_header = ["label_" + str(i) for i in range(self.n_outputs)]
        self.attributes_header = ["att_num_" + str(i) for i in range(self.num_numerical_attributes)]

    def estimated_remaining_instances(self):
        return self.n_samples - self.instance_index

    def has_more_instances(self):
        return self.n_samples - self.instance_index > 0

    def next_instance(self, batch_size=1):
        """ next_instance
        
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
        self.instance_index += batch_size
        try:
            self.current_instance_x = self.X[self.instance_index - batch_size:self.instance_index, :]
            self.current_instance_y = self.y[self.instance_index - batch_size:self.instance_index, :]
            if self.n_classes < 2:
                self.current_instance_y = self.current_instance_y.flatten()

        except IndexError:
            self.current_instance_x = None
            self.current_instance_y = None

        return self.current_instance_x, self.current_instance_y

    def is_restartable(self):
        return True

    def restart(self):
        self.instance_index = 0
        self.current_instance_x = None
        self.current_instance_y = None

    def get_num_nominal_attributes(self):
        return self.num_nominal_attributes

    def get_num_numerical_attributes(self):
        return self.num_numerical_attributes

    def get_num_values_per_nominal_attribute(self):
        return self.num_values_per_nominal_att

    def get_num_attributes(self):
        return self.n_features

    def get_num_classes(self):
        return self.n_classes

    def get_attributes_header(self):
        return self.attributes_header

    def get_classes_header(self):
        return self.class_header

    def get_last_instance(self):
        return self.current_instance_x, self.current_instance_y

    def prepare_for_use(self):
        pass

    def get_plot_name(self):
        return 'Multilabel Generator'

    def get_classes(self):
        return np.unique(self.y).tolist()

    def get_class_type(self):
        return 'stream'

    def get_info(self):
        return 'MultilabelGenerator: n_samples: ' + str(self.n_samples) + \
               ' - n_features: ' + str(self.n_features) + \
               ' - n_targets: ' + str(self.n_classes) + \
               ' - n_labels:' + str(self.n_outputs)

    def get_num_outputs(self):
        return self.n_outputs
