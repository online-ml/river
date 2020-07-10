import numpy as np
from skmultiflow.data.base_stream import Stream
from skmultiflow.utils import check_random_state


class STAGGERGenerator(Stream):
    """ STAGGER concepts stream generator.

    This generator is an implementation of the dara stream with abrupt concept
    drift, as described in Gama, Joao, et al [1]_.

    The STAGGER Concepts are boolean functions f three features encoding
    objects: size (small, medium and large), shape (circle, square and
    triangle) and colour (red, blue and green).
    A classification function is chosen among three possible ones:

    0. Function that return 1 if the size is small and the color is red.
    1. Function that return 1 if the color is green or the shape is a circle.
    2. Function that return 1 if the size is medium or large

    Concept drift can be introduced by changing the classification function.
    This can be done manually or using ``ConceptDriftStream``.

    One important feature is the possibility to balance classes, which
    means the class distribution will tend to a uniform one.

    Parameters
    ----------
    classification_function: int (Default: 0)
        Which of the four classification functions to use for the generation.
        The value can vary from 0 to 2.

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by `np.random`.

    balance_classes: bool (Default: False)
        Whether to balance classes or not. If balanced, the class distribution
        will converge to a uniform distribution.

    References
    ----------
    .. [1] Gama, Joao, et al.'s 'Learning with drift detection.
       ' Advances in artificial intelligence–SBIA 2004. Springer Berlin
       Heidelberg, 2004. 286-295."


    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.data.stagger_generator import STAGGERGenerator
    >>> # Setting up the stream
    >>> stream = STAGGERGenerator(classification_function = 2, random_state = 112,
    ...  balance_classes = False)
    >>> # Retrieving one sample
    >>> stream.next_sample()
    (array([[0., 0., 2.]]), array([0.]))
    >>> stream.next_sample(10)
    (array([[1., 0., 1.],
       [0., 0., 0.],
       [1., 2., 0.],
       [1., 0., 2.],
       [0., 2., 1.],
       [0., 1., 2.],
       [0., 1., 1.],
       [0., 1., 2.],
       [1., 2., 2.],
       [1., 2., 0.]]), array([1., 0., 1., 1., 0., 0., 0., 0., 1., 1.]))
    >>> stream.n_remaining_samples()
    -1
    >>> stream.has_more_samples()
    True

    """

    def __init__(self, classification_function=0, random_state=None, balance_classes=False):
        super().__init__()

        # Classification functions to use
        self._classification_functions = [self.classification_function_zero,
                                          self.classification_function_one,
                                          self.classification_function_two]

        self.classification_function = classification_function
        self.random_state = random_state
        self.balance_classes = balance_classes
        self.n_cat_features = 3
        self.n_features = self.n_cat_features
        self.n_classes = 2
        self.n_targets = 1
        self._random_state = None  # This is the actual random_state object used internally
        self.next_class_should_be_zero = False
        self.name = "Stagger Generator"

        self.target_names = ["target_0"]
        self.feature_names = ["size", "color", "shape"]
        self.size_labels = ["small", "medium", "large"]
        self.color_labels = ["red", "blue", "green"]
        self.shape_labels = ["circle", "square", "triangle"]
        self.target_values = [i for i in range(self.n_classes)]

        self._prepare_for_use()

    @property
    def classification_function(self):
        """ Retrieve the index of the current classification function.

        Returns
        -------
        int
            index of the classification function from 0 to 2.
        """
        return self._classification_function_idx

    @classification_function.setter
    def classification_function(self, classification_function_idx):
        """ Set the index of the current classification function.

        Parameters
        ----------
        classification_function_idx: int (0,1,2)
        """
        if classification_function_idx in range(3):
            self._classification_function_idx = classification_function_idx
        else:
            raise ValueError("classification_function takes only these "
                             "values: 0, 1, 2, and {} was "
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
            raise ValueError(
                "balance_classes should be boolean, and {} was passed".format(balance_classes))

    def _prepare_for_use(self):
        self._random_state = check_random_state(self.random_state)
        self.next_class_should_be_zero = False

    def next_sample(self, batch_size=1):
        """ Returns next sample from the stream.

        The sample generation works as follows: The three attributes are
        generated with the random int generator, initialized with the seed
        passed by the user. Then, the classification function decides whether
        to classify the instance as class 0 or class 1. The next step is to
        verify if the classes should be balanced, and if so, balance the
        classes.

        The generated sample will have  relevant features and 1 label (it has
        one classification task).

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
            size = color = shape = 0
            group = 0
            desired_class_found = False
            while not desired_class_found:
                size = self._random_state.randint(3)
                color = self._random_state.randint(3)
                shape = self._random_state.randint(3)

                group = self._classification_functions[self.classification_function](size, color,
                                                                                     shape)

                if not self.balance_classes:
                    desired_class_found = True
                else:
                    if (self.next_class_should_be_zero and (group == 0)) or \
                            ((not self.next_class_should_be_zero) and (group == 1)):
                        desired_class_found = True
                        self.next_class_should_be_zero = not self.next_class_should_be_zero

            data[j, 0] = size
            data[j, 1] = color
            data[j, 2] = shape
            data[j, 3] = group

        self.current_sample_x = data[:, :self.n_features]
        self.current_sample_y = data[:, self.n_features:].flatten().astype(int)

        return self.current_sample_x, self.current_sample_y

    def generate_drift(self):
        """
        Generate drift by switching the classification function randomly.

        """
        new_function = self._random_state.randint(3)
        while new_function == self.classification_function:
            new_function = self._random_state.randint(3)
        self.classification_function = new_function

    @staticmethod
    def classification_function_zero(size, color, shape):
        """ classification_function_zero

        Decides the sample class label as positive if the color is red and
        size is small.

        Parameters
        ----------
        size: int
            First numeric attribute.

        color: int
            Second boolean attribute.

        shape: int
            Third boolean attribute

        Returns
        -------
            int
            Returns the sample class label, either 0 or 1.

               """
        return 1 if (size == 0 and color == 0) else 0

    @staticmethod
    def classification_function_one(size, color, shape):
        """ classification_function_one

        Decides the sample class label as positive if the color is green or
        shape is a circle.

        Parameters
        ----------
        size: int
            First numeric attribute.

        color: int
            Second boolean attribute.

        shape: int
            Third boolean attribute

        Returns
        -------
        int
            Returns the sample class label, either 0 or 1.

        """
        return 1 if (color == 2 or shape == 0) else 0

    @staticmethod
    def classification_function_two(size, color, shape):
        """ classification_function_two

        Decides the sample class label as positive if the size is medium or
        large.

        Parameters
        ----------
        size: int
            First numeric attribute.

        color: int
            Second boolean attribute.

        shape: int
            Third boolean attribute

        Returns
        -------
        int
            Returns the sample class label, either 0 or 1.

        """
        return 1 if (size == 1 or size == 2) else 0
