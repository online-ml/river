import numpy as np
from skmultiflow.data.base_stream import Stream
from skmultiflow.utils import check_random_state


class AGRAWALGenerator(Stream):
    """ AGRAWAL stream generator

    The generator was introduced by Agrawal et al in [1]_, and was common source
    of data for early work on scaling up decision tree learners.
    The generator produces a stream containing nine features, six numeric and
    three categorical.
    There are ten functions defined for generating binary class labels from the
    features. Presumably these determine whether the loan should be approved.
    The features and functions are listed in the original paper [1]_.

    Parameters
    ----------
    classification_function: int (Default=0)
        Which of the four classification functions to use for the generation.
        The value can vary from 0 to 9.

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    balance_classes: bool (Default: False)
        Whether to balance classes or not. If balanced, the class
        distribution will converge to a uniform distribution.

    perturbation: float (Default: 0.0)
        The probability that noise will happen in the generation. At each
        new sample generated, the sample with will perturbed by the amount of
        perturbation.
        Values go from 0.0 to 1.0.

    References
    ---------
    .. [1] Rakesh Agrawal, Tomasz Imielinksi, and Arun Swami. "Database
       Mining: A Performance Perspective", IEEE Transactions on Knowledge and
       Data Engineering, 5(6), December 1993.

    """
    def __init__(self, classification_function=0, random_state=None, balance_classes=False, perturbation=0.0):
        super().__init__()

        # Classification functions to use
        self._classification_functions = [self.classification_function_zero, self.classification_function_one,
                                          self.classification_function_two, self.classification_function_three,
                                          self.classification_function_four, self.classification_function_five,
                                          self.classification_function_six, self.classification_function_seven,
                                          self.classification_function_eight, self.classification_function_nine]
        self.classification_function_idx = classification_function
        self._original_random_state = random_state
        self.balance_classes = balance_classes
        self.perturbation = perturbation
        self.n_num_features = 6
        self.n_cat_features = 3
        self.n_features = self.n_num_features + self.n_cat_features
        self.n_classes = 2
        self.n_targets = 1
        self.random_state = None
        self._next_class_should_be_zero = False
        self.name = "AGRAWAL Generator"

        self.__configure()

    def __configure(self):
        self.target_names = ["target"]
        self.feature_names = ["salary", "commission", "age", "elevel", "car", "zipcode", "hvalue", "hyears", "loan"]
        self.target_values = [i for i in range(self.n_classes)]

    @property
    def classification_function_idx(self):
        """ Retrieve the index of the current classification function.

        Returns
        -------
        int
            index of the classification function, from 0 to 9
        """
        return self._classification_function_idx

    @classification_function_idx.setter
    def classification_function_idx(self, classification_function_idx):
        """ Set the index of the current classification function.

        Parameters
        ----------
        classification_function_idx: int
            from 0 to 9
        """
        if classification_function_idx in range(10):
            self._classification_function_idx = classification_function_idx
        else:
            raise ValueError("classification_function_idx takes values from 0 to 9,"
                             " and {} was passed".format(classification_function_idx))

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
            raise ValueError("balance_classes should be boolean, and {} was passed".format(balance_classes))

    @property
    def perturbation(self):
        """ Retrieve the value of the option: Noise percentage

        Returns
        -------
        float

        """
        return self._perturbation

    @perturbation.setter
    def perturbation(self, perturbation):
        """ Set the value of the option: Perturbation.

        Parameters
        ----------
        perturbation: float
            from 0.0 to 1.0.

        """
        if (0.0 <= perturbation) and (perturbation <= 1.0):
            self._perturbation = perturbation
        else:
            raise ValueError("noise percentage should be in [0.0..1.0], and {} was passed".format(perturbation))

    def prepare_for_use(self):
        """
        Should be called before generating the samples.

        """
        self.random_state = check_random_state(self._original_random_state)
        self._next_class_should_be_zero = False
        self.sample_idx = 0

    def next_sample(self, batch_size=1):
        """ next_sample

        The sample generation works as follows: The 9 features are generated
        with the random generator, initialized with the seed passed by the
        user. Then, the classification function decides, as a function of all
        the attributes, whether to classify the instance as class 0 or class
        1. The next step is to verify if the classes should be balanced, and
        if so, balance the classes. The last step is to add noise, if the noise
        percentage is higher than 0.0.

        The generated sample will have 9 features and 1 label (it has one
        classification task).

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
        data = np.zeros([batch_size, self.n_features + 1])

        for j in range(batch_size):
            self.sample_idx += 1
            salary = commission = age = elevel = car = zipcode = hvalue = hyears = loan = 0
            group = 0
            desired_class_found = False
            while not desired_class_found:
                salary = 20000 + 130000 * self.random_state.rand()
                commission = 0 if (salary >= 75000) else (10000 + 75000 * self.random_state.rand())
                age = 20 + self.random_state.randint(61)
                elevel = self.random_state.randint(5)
                car = self.random_state.randint(20)
                zipcode = self.random_state.randint(9)
                hvalue = (9 - zipcode) * 100000 * (0.5 + self.random_state.rand())
                hyears = 1 + self.random_state.randint(30)
                loan = self.random_state.rand() * 100000
                group = self._classification_functions[self.classification_function_idx](salary, commission,
                                                                                         age, elevel, car,
                                                                                         zipcode, hvalue,
                                                                                         hyears, loan)
                if not self.balance_classes:
                    desired_class_found = True
                else:
                    if (self._next_class_should_be_zero and (group == 0)) or \
                            ((not self._next_class_should_be_zero) and (group == 1)):
                        desired_class_found = True
                        self._next_class_should_be_zero = not self._next_class_should_be_zero

            if self.perturbation > 0.0:
                salary = self.perturb_value(salary, 20000, 150000)
                if commission > 0:
                    commission = self.perturb_value(commission, 10000, 75000)
                age = np.round(self.perturb_value(age, 20, 80))
                hvalue = self.perturb_value(hvalue, (9 - zipcode) * 100000, 0, 135000)
                hyears = np.round(self.perturb_value(hyears, 1, 30))
                loan = self.perturb_value(loan, 0, 500000)

            for i in range(9):
                data[j, i] = eval(self.feature_names[i])
            data[j, 9] = group

        self.current_sample_x = data[:, :self.n_features]
        self.current_sample_y = data[:, self.n_features:].flatten()

        return self.current_sample_x, self.current_sample_y

    def perturb_value(self, val, val_min, val_max, val_range=None):
        """
        Perturbs the values of the features after prediction the class
        if the perturbation is higher than 0.0.
        Parameters
        ----------
        val
        val_min
        val_max
        val_range

        Returns
        -------
        val
            The value after perturbation.
        """
        if val_range is None:
            val_range = val_max - val_min
        val += val_range * (2 * (self.random_state.rand() - 0.5)) * self.perturbation
        if val < val_min:
            val = val_min
        elif val > val_max:
            val = val_max
        return val

    def generate_drift(self):
        """
        Generate drift by switching the classification function randomly.

        """
        new_function = self.random_state.randint(10)
        while new_function == self.classification_function_idx:
            new_function = self.random_state.randint(10)
        self.classification_function_idx = new_function


    @staticmethod
    def classification_function_zero(salary, commission, age, elevel, car, zipcode, hvalue, hyears, loan):
        """ classification_function_zero

        Parameters
        ----------
        age: float
            First numeric attribute.

        commission: float
            Second numeric attribute.

        age: int
            Third numeric attribute.

        elevel: int
            Forth numeric attribute.

        car: int
            fifth numeric attribute.

        zipcode; int
            sixth numeric attribute.

        hvalue: flaot
            seventh numeric attribute.

        hyears: float
            eighth numeric attribute.

        loan: float
            ninth numeric attribute.

        Returns
        -------
        int
            Returns the sample class label, either 0 or 1.

        """

        return 0 if ((age < 40) or (60 <= age)) else 1

    @staticmethod
    def classification_function_one(salary, commission, age, elevel, car, zipcode, hvalue, hyears, loan):
        """ classification_function_one

        Parameters
        ----------
        age: float
            First numeric attribute.

        commission: float
            Second numeric attribute.

        age: int
            Third numeric attribute.

        elevel: int
            Forth numeric attribute.

        car: int
            fifth numeric attribute.

        zipcode; int
            sixth numeric attribute.

        hvalue: flaot
            seventh numeric attribute.

        hyears: float
            eighth numeric attribute.

        loan: float
            ninth numeric attribute.

        Returns
        -------
        int
            Returns the sample class label, either 0 or 1.

        """

        if age < 40:
            return 0 if ((50000 <= salary) and (salary <= 100000)) else 1
        elif age < 60:
            return 0 if ((75000 <= salary) and (salary <= 125000)) else 1
        else:
            return 0 if ((25000 <= salary) and (salary <= 75000)) else 1

    @staticmethod
    def classification_function_two(salary, commission, age, elevel, car, zipcode, hvalue, hyears, loan):
        """ classification_function_two

        Parameters
        ----------
        age: float
            First numeric attribute.

        commission: float
            Second numeric attribute.

        age: int
            Third numeric attribute.

        elevel: int
            Forth numeric attribute.

        car: int
            fifth numeric attribute.

        zipcode; int
            sixth numeric attribute.

        hvalue: flaot
            seventh numeric attribute.

        hyears: float
            eighth numeric attribute.

        loan: float
            ninth numeric attribute.

        Returns
        -------
        int
            Returns the sample class label, either 0 or 1.

        """

        if age < 40:
            return 0 if ((elevel == 0) or (elevel == 1)) else 1
        elif age < 60:
            return 0 if ((elevel == 1) or (elevel == 2) or (elevel == 3)) else 1
        else:
            return 0 if ((elevel == 2) or (elevel == 3)) or (elevel == 4) else 1

    @staticmethod
    def classification_function_three(salary, commission, age, elevel, car, zipcode, hvalue, hyears, loan):
        """ classification_function_three

        Parameters
        ----------
        age: float
            First numeric attribute.

        commission: float
            Second numeric attribute.

        age: int
            Third numeric attribute.

        elevel: int
            Forth numeric attribute.

        car: int
            fifth numeric attribute.

        zipcode; int
            sixth numeric attribute.

        hvalue: flaot
            seventh numeric attribute.

        hyears: float
            eighth numeric attribute.

        loan: float
            ninth numeric attribute.

        Returns
        -------
        int
            Returns the sample class label, either 0 or 1.

        """

        if age < 40:
            if (elevel == 0) or (elevel == 1):
                return 0 if ((25000 <= salary) and (salary <= 75000)) else 1
            else:
                return 0 if ((50000 <= salary) and (salary <= 100000)) else 1
        elif age < 60:
            if (elevel == 1) or (elevel == 2) or (elevel == 3):
                return 0 if ((50000 <= salary) and (salary <= 100000)) else 1
            else:
                return 0 if ((75000 <= salary) and (salary <= 125000)) else 1
        else:
            if (elevel == 2) or (elevel == 3) or (elevel == 4):
                return 0 if ((50000 <= salary) and (salary <= 100000)) else 1
            else:
                return 0 if ((25000 <= salary) and (salary <= 75000)) else 1

    @staticmethod
    def classification_function_four(salary, commission, age, elevel, car, zipcode, hvalue, hyears, loan):
        """ classification_function_four

        Parameters
        ----------
        age: float
            First numeric attribute.

        commission: float
            Second numeric attribute.

        age: int
            Third numeric attribute.

        elevel: int
            Forth numeric attribute.

        car: int
            fifth numeric attribute.

        zipcode; int
            sixth numeric attribute.

        hvalue: flaot
            seventh numeric attribute.

        hyears: float
            eighth numeric attribute.

        loan: float
            ninth numeric attribute.

        Returns
        -------
        int
            Returns the sample class label, either 0 or 1.

        """

        if age < 40:
            if (50000 <= salary) and (salary <= 100000):
                return 0 if ((100000 <= loan) and (loan <= 300000)) else 1
            else:
                return 0 if ((200000 <= salary) and (salary <= 400000)) else 1
        elif age < 60:
            if (75000 <= salary) and (salary <= 125000):
                return 0 if ((200000 <= salary) and (loan <= 400000)) else 1
            else:
                return 0 if ((300000 <= salary) and (salary <= 500000)) else 1
        else:
            if (25000 <= salary) and (salary <= 75000):
                return 0 if ((300000 <= loan) and (loan <= 500000)) else 1
            else:
                return 0 if ((75000 <= loan) and (loan <= 300000)) else 1

    @staticmethod
    def classification_function_five(salary, commission, age, elevel, car, zipcode, hvalue, hyears, loan):
        """ classification_function_five

        Parameters
        ----------
        age: float
            First numeric attribute.

        commission: float
            Second numeric attribute.

        age: int
            Third numeric attribute.

        elevel: int
            Forth numeric attribute.

        car: int
            fifth numeric attribute.

        zipcode; int
            sixth numeric attribute.

        hvalue: flaot
            seventh numeric attribute.

        hyears: float
            eighth numeric attribute.

        loan: float
            ninth numeric attribute.

        Returns
        -------
        int
            Returns the sample class label, either 0 or 1.

        """

        totalsalary = salary + commission

        if age < 40:
            return 0 if ((50000 <= totalsalary) and (totalsalary <= 100000)) else 1
        elif age < 60:
            return 0 if ((75000 <= totalsalary) and (totalsalary <= 125000)) else 1
        else:
            return 0 if ((25000 <= totalsalary) and (totalsalary <= 75000)) else 1

    @staticmethod
    def classification_function_six(salary, commission, age, elevel, car, zipcode, hvalue, hyears, loan):
        """ classification_function_six

        Parameters
        ----------
        age: float
            First numeric attribute.

        commission: float
            Second numeric attribute.

        age: int
            Third numeric attribute.

        elevel: int
            Forth numeric attribute.

        car: int
            fifth numeric attribute.

        zipcode; int
            sixth numeric attribute.

        hvalue: flaot
            seventh numeric attribute.

        hyears: float
            eighth numeric attribute.

        loan: float
            ninth numeric attribute.

        Returns
        -------
        int
            Returns the sample class label, either 0 or 1.

        """
        disposable = (2 * (salary + commission) / 3 - loan / 5 - 20000)
        return 0 if disposable > 1 else 1

    @staticmethod
    def classification_function_seven(salary, commission, age, elevel, car, zipcode, hvalue, hyears, loan):
        """ classification_function_seven

        Parameters
        ----------
        age: float
            First numeric attribute.

        commission: float
            Second numeric attribute.

        age: int
            Third numeric attribute.

        elevel: int
            Forth numeric attribute.

        car: int
            fifth numeric attribute.

        zipcode; int
            sixth numeric attribute.

        hvalue: flaot
            seventh numeric attribute.

        hyears: float
            eighth numeric attribute.

        loan: float
            ninth numeric attribute.

        Returns
        -------
        int
            Returns the sample class label, either 0 or 1.

        """
        disposable = (2 * (salary + commission) / 3 - 5000 * elevel - 20000)
        return 0 if disposable > 1 else 1

    @staticmethod
    def classification_function_eight(salary, commission, age, elevel, car, zipcode, hvalue, hyears, loan):
        """ classification_function_eight

        Parameters
        ----------
        age: float
            First numeric attribute.

        commission: float
            Second numeric attribute.

        age: int
            Third numeric attribute.

        elevel: int
            Forth numeric attribute.

        car: int
            fifth numeric attribute.

        zipcode; int
            sixth numeric attribute.

        hvalue: flaot
            seventh numeric attribute.

        hyears: float
            eighth numeric attribute.

        loan: float
            ninth numeric attribute.

        Returns
        -------
        int
            Returns the sample class label, either 0 or 1.

        """
        disposable = (2 * (salary + commission) / 3 - 5000 * elevel - loan / 5 - 10000)
        return 0 if disposable > 1 else 1

    @staticmethod
    def classification_function_nine(salary, commission, age, elevel, car, zipcode, hvalue, hyears, loan):
        """ classification_function_nine

        Parameters
        ----------
        age: float
            First numeric attribute.

        commission: float
            Second numeric attribute.

        age: int
            Third numeric attribute.

        elevel: int
            Forth numeric attribute.

        car: int
            fifth numeric attribute.

        zipcode; int
            sixth numeric attribute.

        hvalue: flaot
            seventh numeric attribute.

        hyears: float
            eighth numeric attribute.

        loan: float
            ninth numeric attribute.

        Returns
        -------
        int
            Returns the sample class label, either 0 or 1.

        """
        equity = 0
        if hyears >= 20:
            equity = hvalue * (hyears - 20) / 10
        disposable = (2 * (salary + commission) / 3 - 5000 * elevel + equity / 5 - 10000)
        return 0 if disposable > 1 else 1

    def get_info(self):
        return 'AGRAWAL Generator: classification_function: ' + str(self.classification_function_idx) + \
               ' - random_state: ' + str(self._original_random_state) + \
               ' - balance_classes: ' + str(self.balance_classes) + \
               ' - perturbation: ' + str(self.perturbation)
