from skmultiflow.core.base_object import BaseObject

class Predicate(BaseObject):
    """ Basic element of a Rule.

    A predicate is a comparison between an attribute and a value in the form of:

        - :math:`Att_{idx} > value`.

    Comparison operators can be:

        - >, >=, < and <= for numeric attributes.
        - == and != for nominal attributes.

    parameters
    ----------
    att_idx: int
        The index of the attribute that is described by the predicate.
    operator: string
        The operator that states the relation between the attribute and the value.
    value: float or int
        The value to which the attribute is compared.

    Note
    ----
    Different forms of predicate can be created by overriding the class' methods.
    """

    def __init__(self, att_idx=None, operator=None, value=None):
        """ Predicate class constructor. """
        self.att_idx = att_idx
        self.operator = operator
        self.value = value

    @property
    def att_idx(self):
        return self._att_idx

    @att_idx.setter
    def att_idx(self, att_idx):
        self._att_idx = att_idx

    @property
    def operator(self):
        return self._operator

    @operator.setter
    def operator(self, operator):
        self._operator = operator

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    def covers_instance(self, X):
        """ Check if the instance X is covered by the predicate.

        parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes to test on the predicate.

        returns
        -------
        Boolean
            True if the predicate covers the instance else False.

        """

        result = False
        if self.operator == ">":
            return X[self.att_idx] > self.value
        elif self.operator == "<":
            return X[self.att_idx] < self.value
        elif self.operator == ">=":
            return X[self.att_idx] >= self.value
        elif self.operator == "<=":
            return X[self.att_idx] <= self.value
        elif self.operator == "==" or self.operator == "=":
            return X[self.att_idx] == self.value
        elif self.operator == "!=":
            return X[self.att_idx] != self.value
        return result

    def get_predicate(self):
        """ Get the predicate

        returns
        -------
        string
            the conjunction described by the predicate.
        """
        return str("Att (" + str(self.att_idx) + ") " + str(self.operator) + " " +
                   "%.3f" % round(self.value, 2))

    def __str__(self):
        """ Print the predicate.

        Returns
        -------
        string
            A string representing the predicate.
        """
        return str("Att (" + str(self.att_idx) + ") " + str(self.operator) + " " +
                   "%.3f" % round(self.value, 2))

    def __eq__(self, other):
        """ Checks is too predicates are equal, meaning have same operator and value.

        Parameters
        ----------
        other: Predicate
            The predicate to compare against.

        Returns
        -------
        Bool
            True if the two predicates are equal.
        """
        if isinstance(other, Predicate):
            if other.att_idx == self.att_idx and other.operator == self.operator:
                return True
        return False

    def get_info(self):
        """ Collect information about the predicate configuration.

        Returns
        -------
        string
            Configuration for the predicate.
        """
        description = self.__class__.__name__ + ': '
        description += 'Attribute index: {} - '.format(self.att_idx)
        description += 'Operator: {} - '.format(self.operator)
        description += 'Comparison value: {}'.format(self.value)
        return description

    def get_class_type(self):
        return 'predicate'