import numpy as np
from skmultiflow.data.base_stream import Stream
from skmultiflow.utils import check_random_state


class SEAGenerator(Stream):
    r""" SEA stream generator.

    This generator is an implementation of the data stream with abrupt
    concept drift, first described in Street and Kim's 'A streaming
    ensemble algorithm (SEA) for large-scale classification' [1]_.

    It generates 3 numerical attributes, that vary from 0 to 10, where
    only 2 of them are relevant to the classification task. A classification
    function is chosen, among four possible ones. These functions compare
    the sum of the two relevant attributes with a threshold value, unique
    for each of the classification functions. Depending on the comparison
    the generator will classify an instance as one of the two possible
    labels.

    The functions are:
     * Function 0: if :math:`(att1 + att2 \leq 8)` else 1
     * Function 1: if :math:`(att1 + att2 \leq 9)` else 1
     * Function 2: if :math:`(att1 + att2 \leq 7)` else 1
     * Function 3: if :math:`(att1 + att2 \leq 9.5)` else 1

    Concept drift can be introduced by changing the classification function.
    This can be done manually or using ``ConceptDriftStream``.

    This data stream has two additional parameters, the first is to balance classes, which
    means the class distribution will tend to a uniform one, and the possibility
    to add noise, which will, according to some probability, change the chosen
    label for an instance.

    Parameters
    ----------
    classification_function: int (Default: 0)
        Which of the four classification functions to use for the generation.
        This value can vary from 0 to 3, and the thresholds are, 8, 9, 7 and 9.5.

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    balance_classes: bool (Default: False)
        Whether to balance classes or not. If balanced, the class
        distribution will converge to a uniform distribution.

    noise_percentage: float (Default: 0.0)
        The probability that noise will happen in the generation. At each
        new sample generated, a random probability is generated, and if that
        probability is higher than the noise_percentage, the chosen label will
        be switched. From 0.0 to 1.0.

    References
    ----------
    .. [1]  W. Nick Street and YongSeog Kim. 2001. A streaming ensemble algorithm (SEA)
       for large-scale classification. In Proceedings of the seventh ACM SIGKDD international
       conference on Knowledge discovery and data mining (KDD '01). ACM, New York, NY, USA,
       377-382. DOI=http://dx.doi.org/10.1145/502512.502568

    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.data.sea_generator import SEAGenerator
    >>> # Setting up the stream
    >>> stream = SEAGenerator(classification_function = 2, random_state = 112,
    ...  balance_classes = False, noise_percentage = 0.28)
    >>> # Retrieving one sample
    >>> stream.next_sample()
    (array([[ 3.75057129,  6.4030462 ,  9.50016579]]), array([ 0.]))
    >>> # Retrieving 10 samples
    >>> stream.next_sample(10)
    (array([[ 7.76929659,  8.32745763,  0.5480574 ],
       [ 8.85351458,  7.22346511,  0.02556032],
       [ 3.43419851,  0.94759888,  3.94642589],
       [ 7.3670683 ,  9.55806869,  8.20609371],
       [ 3.78544458,  7.84763615,  0.86231513],
       [ 1.6222602 ,  2.90069726,  0.45008172],
       [ 7.36533216,  8.39211485,  7.09361615],
       [ 9.8566856 ,  3.88003308,  5.03154482],
       [ 6.8373245 ,  7.21957381,  2.14152091],
       [ 0.75216155,  6.10890702,  4.25630425]]),
       array([ 1.,  1.,  1.,  1.,  1.,  0.,  0.,  1.,  1.,  1.]))
    >>> # Generators will have infinite remaining instances, so it returns -1
    >>> stream.n_remaining_samples()
    -1
    >>> stream.has_more_samples()
    True

    """

    def __init__(self, classification_function=0, random_state=None, balance_classes=False,
                 noise_percentage=0.0):
        super().__init__()

        # Classification functions to use
        self._classification_functions = [self._classification_function_zero,
                                          self._classification_function_one,
                                          self._classification_function_two,
                                          self._classification_function_three]

        self.classification_function = classification_function
        self.random_state = random_state
        self.balance_classes = balance_classes
        self.noise_percentage = noise_percentage
        self.n_num_features = 3
        self.n_features = self.n_num_features
        self.n_classes = 2
        self.n_targets = 1
        self._random_state = None  # This is the actual random_state object used internally
        self.next_class_should_be_zero = False
        self.name = "SEA Generator"

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
            raise ValueError("classification_function takes only these values: 0, 1, 2, 3, {} was "
                             "passed".format(classification_function_idx))

    @property
    def balance_classes(self):
        """ Retrieve the value of the option: Balance classes.

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
                "balance_classes should be boolean, {} was passed".format(balance_classes))

    @property
    def noise_percentage(self):
        """ Retrieve the value of the value of Noise percentage

        Returns
        -------
        float
            percentage of the noise
        """
        return self._noise_percentage

    @noise_percentage.setter
    def noise_percentage(self, noise_percentage):
        """ Set the value of the value of noise percentage.

        Parameters
        ----------
        noise_percentage: float (0.0..1.0)

        """
        if (0.0 <= noise_percentage) and (noise_percentage <= 1.0):
            self._noise_percentage = noise_percentage
        else:
            raise ValueError(
                "noise percentage should be in [0.0..1.0], {} was passed".format(noise_percentage))

    def _prepare_for_use(self):
        self._random_state = check_random_state(self.random_state)
        self.next_class_should_be_zero = False

    def next_sample(self, batch_size=1):
        """ Returns next sample from the stream.

        The sample generation works as follows: The three attributes are
        generated with the random generator, initialized with the seed passed
        by the user. Then, the classification function decides, as a function
        of the two relevant attributes, whether to classify the instance as
        class 0 or class 1. The next step is to verify if the classes should
        be balanced, and if so, balance the classes. The last step is to add
        noise, if the noise percentage is higher than 0.0.

        The generated sample will have 3 features, where only the two first
        are relevant, and 1 label (it has one classification task).

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
            att1 = att2 = att3 = 0.0
            group = 0
            desired_class_found = False
            while not desired_class_found:
                att1 = 10 * self._random_state.rand()
                att2 = 10 * self._random_state.rand()
                att3 = 10 * self._random_state.rand()
                group = self._classification_functions[self.classification_function](att1, att2,
                                                                                     att3)

                if not self.balance_classes:
                    desired_class_found = True
                else:
                    if (self.next_class_should_be_zero and (group == 0)) or \
                            ((not self.next_class_should_be_zero) and (group == 1)):
                        desired_class_found = True
                        self.next_class_should_be_zero = not self.next_class_should_be_zero

            if 0.01 + self._random_state.rand() <= self.noise_percentage:
                group = 1 if (group == 0) else 0

            data[j, 0] = att1
            data[j, 1] = att2
            data[j, 2] = att3
            data[j, 3] = group

        self.current_sample_x = data[:, :self.n_features]
        self.current_sample_y = data[:, self.n_features:].flatten().astype(np.int64)

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
    def _classification_function_zero(att1, att2, att3):
        """ classification_function_zero

        Decides the sample class label based on the sum of att1 and att2,
        and the threshold value of 8.

        Parameters
        ----------
        att1: float
            First numeric attribute.

        att2: float
            Second numeric attribute.

        att3: float
            Third numeric attribute.

        Returns
        -------
        int
            Returns the sample class label, either 0 or 1.

        """
        return 0 if (att1 + att2 <= 8) else 1

    @staticmethod
    def _classification_function_one(att1, att2, att3):
        """ classification_function_one

        Decides the sample class label based on the sum of att1 and att2,
        and the threshold value of 9.

        Parameters
        ----------
        att1: float
            First numeric attribute.

        att2: float
            Second numeric attribute.

        att3: float
            Third numeric attribute.

        Returns
        -------
        int
            Returns the sample class label, either 0 or 1.

        """
        return 0 if (att1 + att2 <= 9) else 1

    @staticmethod
    def _classification_function_two(att1, att2, att3):
        """ classification_function_two

        Decides the sample class label based on the sum of att1 and att2,
        and the threshold value of 7.

        Parameters
        ----------
        att1: float
            First numeric attribute.

        att2: float
            Second numeric attribute.

        att3: float
            Third numeric attribute.

        Returns
        -------
        int
            Returns the sample class label, either 0 or 1.

        """
        return 0 if (att1 + att2 <= 7) else 1

    @staticmethod
    def _classification_function_three(att1, att2, att3):
        """ classification_function_three

        Decides the sample class label based on the sum of att1 and att2,
        and the threshold value of 9.5.

        Parameters
        ----------
        att1: float
            First numeric attribute.

        att2: float
            Second numeric attribute.

        att3: float
            Third numeric attribute.

        Returns
        -------
        int
            Returns the sample class label, either 0 or 1.

        """
        return 0 if (att1 + att2 <= 9.5) else 1
