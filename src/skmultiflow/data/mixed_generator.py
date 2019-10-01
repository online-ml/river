import numpy as np
from skmultiflow.data.base_stream import Stream
from skmultiflow.utils import check_random_state


class MIXEDGenerator(Stream):
    r""" Mixed data stream generator.

    This generator is an implementation of a data stream with abrupt concept drift and boolean noise-free examples
    as described in Gama, João, et al [1]_.

    It has four relevant attributes, two boolean attributes :math:`v, w` and two numeric attributes :math:`x, y`
    uniformly distributed from 0 to 1. The examples are labaled depending on the classification function chosen
    from below.

    * function 0:
        if :math:`v` and :math:`w` are true or :math:`v` and :math:`z` are true or :math:`w` and :math:`z` are
        true then 0 else 1, where :math:`z` is :math:`y < 0.5 + 0.3 sin(3 \pi  x)`
    * function 1:
        The opposite of function 0.

    Concept drift can be introduced by changing the classification function.
    This can be done manually or using ``ConceptDriftStream``.

    Parameters
    ----------
    classification_function: int (default: 0)
        Which of the two classification functions to use for the generation.
        Valid options are 0 or 1.

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    balance_classes: bool (Default: False)
        Whether to balance classes or not. If balanced, the class distribution
        will converge to a uniform distribution.

    References
    ----------
    .. [1] Gama, Joao, et al. "Learning with drift detection." Advances in
       artificial intelligence–SBIA 2004. Springer Berlin Heidelberg,
       2004. 286-295"

    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.data.mixed_generator import MIXEDGenerator
    >>> # Setting up the stream
    >>> stream = MIXEDGenerator(classification_function = 1, random_state= 112, balance_classes = False)
    >>> stream.prepare_for_use()
    >>> # Retrieving one sample
    >>> stream.next_sample()
    (array([[0.        , 1.        , 0.95001658, 0.0756772 ]]), array([1.]))

    >>> stream.next_sample(10)
    (array([[1.        , 1.        , 0.05480574, 0.81767738],
           [1.        , 1.        , 0.00255603, 0.98119928],
           [0.        , 0.        , 0.39464259, 0.00494492],
           [1.        , 1.        , 0.82060937, 0.344983  ],
           [0.        , 1.        , 0.08623151, 0.54607394],
           [0.        , 0.        , 0.04500817, 0.33218776],
           [1.        , 1.        , 0.70936161, 0.18840112],
           [1.        , 0.        , 0.50315448, 0.76353033],
           [1.        , 1.        , 0.21415209, 0.76309258],
           [0.        , 1.        , 0.42563042, 0.23435109]]), array([1., 1., 0., 1., 1., 0., 1., 0., 1., 1.]))

    >>> stream.n_remaining_samples()
    -1
    >>> stream.has_more_samples()
    True

   """

    def __init__(self, classification_function=0, random_state=None, balance_classes=False):
        super().__init__()

        # Classification functions to use
        self._classification_functions = [self._classification_function_zero, self._classification_function_one]
        self.random_state = random_state
        self.classification_function = classification_function
        self._random_state = None   # This is the actual random_state object used internally
        self.balance_classes = balance_classes
        self.n_cat_features = 2
        self.n_num_features = 2
        self.n_features = self.n_cat_features + self.n_num_features
        self.cat_features_idx = [0, 1]
        self.n_classes = 2
        self.n_targets = 1
        self.next_class_should_be_zero = False
        self.name = "Mixed Generator"

        self.__configure()

    def __configure(self):

        self.target_names = ["target_0"]
        self.feature_names = ["att_num_" + str(i) for i in range(self.n_features)]
        self.target_values = [i for i in range(self.n_classes)]

    @property
    def classification_function(self):
        """ Retrieve the index of the current classification function.

        Returns
        -------
        int
            index of the classification function [0,1]
        """
        return self._classification_function_idx

    @classification_function.setter
    def classification_function(self, classification_function_idx):
        """ Set the index of the current classification function.

        Parameters
        ----------
        classification_function_idx: int (0..1)
        """
        if classification_function_idx in range(2):
            self._classification_function_idx = classification_function_idx
        else:
            raise ValueError("classification_function takes only these values: 0, 1, and {} was passed".
                             format(classification_function_idx))

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
            raise ValueError("balance_classes should be boolean, {} was passed".format(balance_classes))

    def next_sample(self, batch_size=1):

        """ next_sample

        The sample generation works as follows: The two numeric attributes are
        generated with the random  generator, initialized with the seed
        passed by the user. The boolean attributes are either 0 or 1
        based on the comparison of the random generator and 0.5 ,
        the classification function decides whether to classify the instance
        as class 0 or class 1. The next step is to verify if the classes should
        be balanced, and if so, balance the classes.

        The generated sample will have 4 relevant features and 1 label (it has
        one classification task).

        Parameters
        ----------
        batch_size: int
            The number of samples to return.

        Returns
        -------
        tuple or tuple list
            Return a tuple with the features matrix and the labels matrix for


                """
        data = np.zeros([batch_size, self.n_features + 1])

        for j in range(batch_size):
            self.sample_idx += 1
            att_0 = att_1 = att_2 = att_3 = 0
            group = 0
            desired_class_found = False
            while not desired_class_found:
                att_0 = 0 if self._random_state.rand() < 0.5 else 1
                att_1 = 0 if self._random_state.rand() < 0.5 else 1
                att_2 = self._random_state.rand()
                att_3 = self._random_state.rand()

                group = self._classification_functions[self.classification_function](att_0, att_1, att_2, att_3)

                if not self.balance_classes:
                    desired_class_found = True
                else:
                    if (self.next_class_should_be_zero and (group == 0)) or \
                            ((not self.next_class_should_be_zero) and (group == 1)):
                        desired_class_found = True
                        self.next_class_should_be_zero = not self.next_class_should_be_zero

            data[j, 0] = att_0
            data[j, 1] = att_1
            data[j, 2] = att_2
            data[j, 3] = att_3
            data[j, 4] = group

        self.current_sample_x = data[:, :self.n_features]
        self.current_sample_y = data[:, self.n_features:].flatten().astype(int)

        return self.current_sample_x, self.current_sample_y

    def prepare_for_use(self):
        """
        Prepares the stream for use.

        Notes
        -----
        This functions should always be called after the stream initialization.

        """
        self._random_state = check_random_state(self.random_state)
        self.next_class_should_be_zero = False
        self.sample_idx = 0

    @staticmethod
    def _classification_function_zero(v, w, x, y):
        """ classification_function_zero

        Decides the sample class label as negative  if the two boolean attributes
        are True or one of them is True and  :math:`y  <  0.5 + 0.3  sin(3  \pi  x)`.

        Parameters
        ----------
        v: boolean
            First boolean attribute.

        w: boolean
            Second boolean attribute.

        x: float
            Third numeric attribute

        y: float
            Third numeric attribute

        Returns
        -------
        int
            Returns the sample class label, either 0 or 1.

        """
        z = y < 0.5 + 0.3 * np.sin(3 * np.pi * x)
        return 0 if (v == 1 and w == 1) or (v == 1 and z) or (w == 1 and z) else 1

    @staticmethod
    def _classification_function_one(v, w, x, y):
        """ classification_function_one

        Decides the sample class label as positive  if the two boolean attributes
        are True or one of them is True and :math:`y < 0.5 + 0.3  sin(3  \pi  x)`.

        Parameters
        ----------
        v: boolean
        First boolean attribute.

        w: boolean
            Second boolean attribute.

        x: float
        Third numeric attribute

        y: float
            Third numeric attribute

        Returns
        -------
        int
            Returns the sample class label, either 0 or 1.

        """
        z = y < 0.5 + 0.3 * np.sin(3 * np.pi * x)
        return 1 if (v == 1 and w == 1) or (v == 1 and z) or (w == 1 and z) else 0

    def generate_drift(self):
        """
        Generate drift by switching the classification function.

        """
        self.classification_function = 1 - self.classification_function
