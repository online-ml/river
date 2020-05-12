import numpy as np
from skmultiflow.data.base_stream import Stream
from skmultiflow.utils import check_random_state


class SineGenerator(Stream):
    r""" Sine stream generator.

    This generator is an implementation of the dara stream with abrupt
    concept drift, as described in Gama, Joao, et al [1]_.

    It generates up to 4 relevant numerical attributes, that vary from 0 to 1,
    where only 2 of them are relevant to the classification task and the other
    2 are added by request of the user. A classification function is chosen
    among four possible ones:

    0. SINE1. Abrupt concept drift, noise-free examples. It has two relevant
       attributes. Each attributes has values uniformly distributed in [0; 1].
       In the first context all points below the curve :math:`y = sin(x)` are
       classified as positive.
    1. Reversed SINE1. The reversed classification of SINE1.
    2. SINE2. The same two relevant attributes. The classification function
       is :math:`y < 0.5 + 0.3 sin(3 \pi  x)`.
    3. Reversed SINE2. The reversed classification of SINE2.

    Concept drift can be introduced by changing the classification function.
    This can be done manually or using ``ConceptDriftStream``.

    Two important features are the possibility to balance classes, which
    means the class distribution will tend to a uniform one, and the possibility
    to add noise, which will, add two non relevant attributes.

    Parameters
    ----------
    classification_function: int (Default: 0)
        Which of the four classification functions to use for the generation.
        From 0 to 3.

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    balance_classes: bool (Default: False)
        Whether to balance classes or not. If balanced, the class
        distribution will converge to a uniform distribution.

    has_noise: bool (Default: False)
        Adds 2 non relevant features to the stream.

    References
    ----------
    .. [1] Gama, Joao, et al.'s 'Learning with drift
       detection.' Advances in artificial intelligence–SBIA 2004. Springer Berlin
       Heidelberg, 2004. 286-295."

    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.data.sine_generator import SineGenerator
    >>> # Setting up the stream
    >>> stream = SineGenerator(classification_function = 2, random_state = 112,
    ...  balance_classes = False, has_noise = True)
    >>> # Retrieving one sample
    >>> stream.next_sample()
    (array([[0.37505713, 0.64030462, 0.95001658, 0.0756772 ]]), array([1.]))
    >>> stream.next_sample(10)
    (array([[0.77692966, 0.83274576, 0.05480574, 0.81767738],
       [0.88535146, 0.72234651, 0.00255603, 0.98119928],
       [0.34341985, 0.09475989, 0.39464259, 0.00494492],
       [0.73670683, 0.95580687, 0.82060937, 0.344983  ],
       [0.37854446, 0.78476361, 0.08623151, 0.54607394],
       [0.16222602, 0.29006973, 0.04500817, 0.33218776],
       [0.73653322, 0.83921149, 0.70936161, 0.18840112],
       [0.98566856, 0.38800331, 0.50315448, 0.76353033],
       [0.68373245, 0.72195738, 0.21415209, 0.76309258],
       [0.07521616, 0.6108907 , 0.42563042, 0.23435109]]),
       array([1., 0., 1., 0., 1., 1., 1., 0., 0., 1.]))
    >>> stream.n_remaining_samples()
    -1
    >>> stream.has_more_samples()
    True

    """
    _NUM_BASE_ATTRIBUTES = 2
    _TOTAL_ATTRIBUTES_INCLUDING_NOISE = 4

    def __init__(self, classification_function=0, random_state=None, balance_classes=False,
                 has_noise=False):
        super().__init__()

        # Classification functions to use
        self._classification_functions = [self._classification_function_zero,
                                          self._classification_function_one,
                                          self._classification_function_two,
                                          self._classification_function_three]
        self.classification_function = classification_function
        self.random_state = random_state
        self.has_noise = has_noise
        self.balance_classes = balance_classes
        self.n_num_features = self._NUM_BASE_ATTRIBUTES
        self.n_classes = 2
        self.n_targets = 1
        self._random_state = None  # This is the actual random_state object used internally
        self.next_class_should_be_zero = False
        self.name = "Sine Generator"

        if self.has_noise:
            self.n_num_features = self._TOTAL_ATTRIBUTES_INCLUDING_NOISE
        self.n_features = self.n_num_features
        self.target_names = ["target_0"]
        self.feature_names = ["att_num_" + str(i) for i in range(self.n_features)]
        self.target_values = [i for i in range(self.n_classes)]

        self._prepare_for_use()

    @property
    def classification_function(self):
        """ Retrieve the index of the current classification function.

        Returns
        -------
        int
            index of the classification function [0,1,2,3]
        """
        return self._classification_function_idx

    @classification_function.setter
    def classification_function(self, classification_function_idx):
        """ Set the index of the current classification function.

        Parameters
        ----------
        classification_function_idx: int (0,1,2,3)
        """
        if classification_function_idx in range(4):
            self._classification_function_idx = classification_function_idx
        else:
            raise ValueError("classification_function takes only these "
                             "values: 0, 1, 2, 3, and {} was "
                             "passed".format(classification_function_idx))

    @property
    def balance_classes(self):
        """ Retrieve the value of the option: Balance classes

        Returns
        -------
        Boolean
            True is the classes are balanced
        """
        return self._balance_classes

    @balance_classes.setter
    def balance_classes(self, balance_classes):
        """ Set the value of the option: Balance classes.

        Parameters
        ----------
        balance_classes: Boolean

        """
        if isinstance(balance_classes, bool):
            self._balance_classes = balance_classes
        else:
            raise ValueError("balance_classes should be boolean,"
                             " and {} was passed".format(balance_classes))

    @property
    def has_noise(self):
        """ Retrieve the value of the option: add noise.

        Returns
        -------
        Boolean
            True is the noise is added
        """
        return self._has_noise

    @has_noise.setter
    def has_noise(self, has_noise):
        """ Set the value of the option: add noise.

        Parameters
        ----------
        has_noise: Boolean

        """
        if isinstance(has_noise, bool):
            self._has_noise = has_noise
        else:
            raise ValueError("has_noise should be boolean, {} was passed".format(has_noise))

    def _prepare_for_use(self):
        self._random_state = check_random_state(self.random_state)
        self.next_class_should_be_zero = False

    def next_sample(self, batch_size=1):
        """ Returns next sample from the stream.

        The sample generation works as follows: The two attributes are
        generated with the random generator, initialized with the seed passed
        by the user. Then, the classification function decides whether to
        classify the instance as class 0 or class 1. The next step is to
        verify if the classes should be balanced, and if so, balance the
        classes. The last step is to add noise, if the has_noise is True.

        The generated sample will have 2 relevant features, and an additional
        two noise features if option chosen, and 1 label (it has one
        classification task).

        Parameters
        ----------
        batch_size: int (optional, default=1)
            The number of samples to return.

        Returns
        -------
        tuple or tuple list
            Return a tuple with the features matrix and the labels matrix for
            the batch_size samples that were requested.

        """

        data = np.zeros([batch_size, self.n_features + 1])

        for j in range(batch_size):
            self.sample_idx += 1
            att1 = att2 = 0.0
            group = 0
            desired_class_found = False
            while not desired_class_found:
                att1 = self._random_state.rand()
                att2 = self._random_state.rand()
                group = self._classification_functions[self.classification_function](att1, att2)

                if not self.balance_classes:
                    desired_class_found = True
                else:
                    if (self.next_class_should_be_zero and (group == 0)) or \
                            ((not self.next_class_should_be_zero) and (group == 1)):
                        desired_class_found = True
                        self.next_class_should_be_zero = not self.next_class_should_be_zero

            data[j, 0] = att1
            data[j, 1] = att2

            if self.has_noise:
                for i in range(self._NUM_BASE_ATTRIBUTES, self._TOTAL_ATTRIBUTES_INCLUDING_NOISE):
                    data[j, i] = self._random_state.rand()
                data[j, 4] = group
            else:
                data[j, 2] = group

        self.current_sample_x = data[:, :self.n_features]
        self.current_sample_y = data[:, self.n_features:].flatten().astype(int)

        return self.current_sample_x, self.current_sample_y

    def generate_drift(self):
        """
        Generate drift by switching the classification function randomly.

        """
        new_function = self._random_state.randint(4)
        while new_function == self.classification_function:
            new_function = self._random_state.randint(4)
        self.classification_function = new_function

    @staticmethod
    def _classification_function_zero(att1, att2):
        """ classification_function_zero

        Decides the sample class label based on SINE1 function.


        Parameters
        ----------
        att1: float
            First numeric attribute.

        att2: float
            Second numeric attribute.

        Returns
        -------
        int
            Returns the sample class label, either 0 or 1.

        """
        return 0 if (att1 >= np.sin(att2)) else 1

    @staticmethod
    def _classification_function_one(att1, att2):
        """ classification_function_one

        Decides the sample class label based on reversed SINE1 function.

        Parameters
        ----------
        att1: float
            First numeric attribute.

        att2: float
            Second numeric attribute.

        Returns
        -------
        int
            Returns the sample class label, either 0 or 1.

        """
        return 0 if (att1 < np.sin(att2)) else 1

    @staticmethod
    def _classification_function_two(att1, att2):
        """ classification_function_two

        Decides the sample class label based on SINE2 function.

        Parameters
        ----------
        att1: float
            First numeric attribute.

        att2: float
            Second numeric attribute.

        Returns
        -------
        int
            Returns the sample class label, either 0 or 1.

        """
        return 0 if (att1 >= 0.5 + 0.3 * np.sin(3 * np.pi * att2)) else 1

    @staticmethod
    def _classification_function_three(att1, att2):
        """ classification_function_three

        Decides the sample class label based on reversed SINE2 function.

        Parameters
        ----------
        att1: float
            First numeric attribute.

        att2: float
            Second numeric attribute.

        Returns
        -------
        int
            Returns the sample class label, either 0 or 1.

        """
        return 0 if (att1 < 0.5 + 0.3 * np.sin(3 * np.pi * att2)) else 1
