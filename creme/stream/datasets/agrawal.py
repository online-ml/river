import random

from . import base
from creme.utils.skmultiflow_utils import check_random_state


class Agrawal(base.SyntheticDataset):
    """Agrawal stream generator.

    The generator was introduced by Agrawal et al. [^1], and was a common
    source of data for early work on scaling up decision tree learners.
    The generator produces a stream containing nine features, six numeric and
    three categorical.
    There are ten functions defined for generating binary class labels from the
    features. Presumably these determine whether the loan should be approved.
    The features and functions are listed in the original paper [^1].

    | feature name | feature description  | values                                                              |
    |--------------|----------------------|---------------------------------------------------------------------|
    | salary       | the salary           | uniformly distributed from 20k to 150k                              |
    | commission   | the commission       | if (salary < 75k) then 0 else uniformly distributed from 10k to 75k |
    | age          | the age              | uniformly distributed from 20 to 80                                 |
    | elevel       | the education level  | uniformly chosen from 0 to 4                                        |
    | car          | car maker            | uniformly chosen from 1 to 20                                       |
    | zipcode      | zip code of the town | uniformly chosen from 0 to 8                                        |
    | hvalue       | value of the house   | uniformly distributed from 50k x zipcode to 100k x zipcode          |
    | hyears       | years house owned    | uniformly distributed from 1 to 30                                  |
    | loan         | total loan amount    | uniformly distributed from 0 to 500k                                |

    Parameters
    ----------
    classification_function
        Which of the four classification functions to use for the generation.
        The value can vary from 0 to 9.
    seed
        Random seed number used for reproducibility.
    balance_classes
        Whether to balance classes or not. If balanced, the class
        distribution will converge to a uniform distribution.
    perturbation
        The probability that noise will happen in the generation. At each
        new sample generated, the sample with will perturbed by the amount of
        perturbation. Vavlid values are in the range from 0.0 to 1.0.

    Examples
    --------

    >>> from creme import stream

    >>> dataset = stream.iter_dataset('Agrawal',
    ...                               classification_function=0,
    ...                               seed=42)

    >>> for x, y in dataset.take(5):
    ...     print(x, y)
    {'salary': 103125.48379952488, 'commission': 0, 'age': 21, 'elevel': 2, 'car': 7, 'zipcode': 3, 'hvalue': 383722.75711508637, 'hyears': 4, 'loan': 338349.74371145567} 1
    {'salary': 135983.3438016299, 'commission': 0, 'age': 25, 'elevel': 4, 'car': 13, 'zipcode': 0, 'hvalue': 476817.4974942633, 'hyears': 7, 'loan': 116330.4466953698} 1
    {'salary': 98262.43477649744, 'commission': 0, 'age': 55, 'elevel': 1, 'car': 17, 'zipcode': 6, 'hvalue': 216132.186612209, 'hyears': 19, 'loan': 139095.35411533137} 0
    {'salary': 133009.0417030814, 'commission': 0, 'age': 68, 'elevel': 1, 'car': 13, 'zipcode': 5, 'hvalue': 311148.53666865674, 'hyears': 7, 'loan': 478606.5361033906} 1
    {'salary': 63757.29086464148, 'commission': 16955.938253511093, 'age': 26, 'elevel': 2, 'car': 11, 'zipcode': 4, 'hvalue': 653564.13663719, 'hyears': 24, 'loan': 229712.43983592128} 1

    References
    ----------
    [^1]: Rakesh Agrawal, Tomasz Imielinksi, and Arun Swami. "Database Mining:
          A Performance Perspective", IEEE Transactions on Knowledge and
          Data Engineering, 5(6), December 1993.

    """
    def __init__(self, classification_function=0,
                 seed: int = None,
                 balance_classes=False,
                 perturbation=0.0):
        super().__init__(n_features=9, n_classes=2, n_outputs=1, task=base.BINARY_CLF)

        # Classification functions to use
        self._classification_functions = [self._classification_function_0,
                                          self._classification_function_1,
                                          self._classification_function_2,
                                          self._classification_function_3,
                                          self._classification_function_4,
                                          self._classification_function_5,
                                          self._classification_function_6,
                                          self._classification_function_7,
                                          self._classification_function_8,
                                          self._classification_function_9]
        if classification_function not in range(10):
            raise ValueError(f"classification_function takes values from 0 to 9 "
                             f"and {classification_function} was passed")
        self.classification_function = classification_function
        self.balance_classes = balance_classes
        if not 0.0 <= perturbation <= 1.0:
            raise ValueError(f"noise percentage should be in [0.0..1.0] "
                             f"and {perturbation} was passed")
        self.perturbation = perturbation
        self.seed = seed
        self._rng = None  # This is the actual random number generator object used internally
        self.n_num_features = 6
        self.n_cat_features = 3
        self._next_class_should_be_zero = False
        self.target_names = ["target"]
        self.feature_names = ["salary", "commission", "age", "elevel", "car", "zipcode", "hvalue",
                              "hyears", "loan"]
        self.target_values = [i for i in range(self.n_classes)]
        # Legacy variables
        self.current_sample_x = None
        self.current_sample_y = None
        self.sample_idx = 0

        self._prepare_for_use()

    def _prepare_for_use(self):
        self._rng = random.Random(self.seed)
        self._next_class_should_be_zero = False

    def restart(self):
        """Restart the stream. """
        self.current_sample_x = None
        self.current_sample_y = None
        self.sample_idx = 0
        self._prepare_for_use()

    def __iter__(self):
        """Generate next sample.

        The sample generation works as follows: The 9 features are generated
        with the random generator, initialized with the seed passed by the
        user. Then, the classification function decides, as a function of all
        the attributes, whether to classify the instance as class 0 or class
        1. The next step is to verify if the classes should be balanced, and
        if so, balance the classes. The last step is to add noise, if the noise
        percentage is higher than 0.0.

        """

        while True:
            self.sample_idx += 1
            group = 0
            desired_class_found = False
            while not desired_class_found:
                salary = 20000 + 130000 * self._rng.random()
                commission = 0 if (salary >= 75000) else (10000 + 75000 * self._rng.random())
                age = 20 + self._rng.randint(0, 60)
                elevel = self._rng.randint(0, 4)
                car = self._rng.randint(0, 19)
                zipcode = self._rng.randint(0, 8)
                hvalue = (9 - zipcode) * 100000 * (0.5 + self._rng.random())
                hyears = 1 + self._rng.randint(0, 29)
                loan = self._rng.random() * 500000
                group = self._classification_functions[self.classification_function](salary,
                                                                                     commission,
                                                                                     age, elevel,
                                                                                     car,
                                                                                     zipcode,
                                                                                     hvalue,
                                                                                     hyears, loan)
                if not self.balance_classes:
                    desired_class_found = True
                else:
                    if (self._next_class_should_be_zero and (group == 0)) or \
                            ((not self._next_class_should_be_zero) and (group == 1)):
                        desired_class_found = True
                        self._next_class_should_be_zero = not self._next_class_should_be_zero

            if self.perturbation > 0.0:
                salary = self._perturb_value(salary, 20000, 150000)
                if commission > 0:
                    commission = self._perturb_value(commission, 10000, 75000)
                age = np.round(self._perturb_value(age, 20, 80))
                hvalue = self._perturb_value(hvalue, (9 - zipcode) * 100000, 0, 135000)
                hyears = np.round(self._perturb_value(hyears, 1, 30))
                loan = self._perturb_value(loan, 0, 500000)

            x = dict()
            for key in self.feature_names:
                x[key] = eval(key)
            y = group

            self.current_sample_x = x
            self.current_sample_y = y

            yield x, y

    def _perturb_value(self, val, val_min, val_max, val_range=None):
        if val_range is None:
            val_range = val_max - val_min
        val += val_range * (2 * (self._rng.rand() - 0.5)) * self.perturbation
        if val < val_min:
            val = val_min
        elif val > val_max:
            val = val_max
        return val

    def generate_drift(self):
        """
        Generate drift by switching the classification function randomly.

        """
        new_function = self._rng.randint(0, 9)
        while new_function == self.classification_function:
            new_function = self._rng.randint(0, 9)
        self.classification_function = new_function

    @staticmethod
    def _classification_function_0(salary, commission, age, elevel, car, zipcode, hvalue,
                                   hyears, loan):
        return int((age < 40) or (60 <= age))

    @staticmethod
    def _classification_function_1(salary, commission, age, elevel, car, zipcode, hvalue, hyears,
                                   loan):
        if age < 40:
            return int((50000 <= salary) and (salary <= 100000))
        elif age < 60:
            return int((75000 <= salary) and (salary <= 125000))
        else:
            return int((25000 <= salary) and (salary <= 75000))

    @staticmethod
    def _classification_function_2(salary, commission, age, elevel, car, zipcode, hvalue, hyears,
                                   loan):
        if age < 40:
            return int((elevel == 0) or (elevel == 1))
        elif age < 60:
            return int((elevel == 1) or (elevel == 2) or (elevel == 3))
        else:
            return int((elevel == 2) or (elevel == 3)) or (elevel == 4)

    @staticmethod
    def _classification_function_3(salary, commission, age, elevel, car, zipcode, hvalue,
                                   hyears, loan):
        if age < 40:
            if (elevel == 0) or (elevel == 1):
                return int((25000 <= salary) and (salary <= 75000))
            else:
                return int((50000 <= salary) and (salary <= 100000))
        elif age < 60:
            if (elevel == 1) or (elevel == 2) or (elevel == 3):
                return int((50000 <= salary) and (salary <= 100000))
            else:
                return int((75000 <= salary) and (salary <= 125000))
        else:
            if (elevel == 2) or (elevel == 3) or (elevel == 4):
                return int((50000 <= salary) and (salary <= 100000))
            else:
                return int((25000 <= salary) and (salary <= 75000))

    @staticmethod
    def _classification_function_4(salary, commission, age, elevel, car, zipcode, hvalue,
                                   hyears, loan):
        if age < 40:
            if (50000 <= salary) and (salary <= 100000):
                return int((100000 <= loan) and (loan <= 300000))
            else:
                return int((200000 <= salary) and (salary <= 400000))
        elif age < 60:
            if (75000 <= salary) and (salary <= 125000):
                return int((200000 <= salary) and (loan <= 400000))
            else:
                return int((300000 <= salary) and (salary <= 500000))
        else:
            if (25000 <= salary) and (salary <= 75000):
                return int((300000 <= loan) and (loan <= 500000))
            else:
                return int((75000 <= loan) and (loan <= 300000))

    @staticmethod
    def _classification_function_5(salary, commission, age, elevel, car, zipcode, hvalue,
                                   hyears, loan):
        totalsalary = salary + commission

        if age < 40:
            return int((50000 <= totalsalary) and (totalsalary <= 100000))
        elif age < 60:
            return int((75000 <= totalsalary) and (totalsalary <= 125000))
        else:
            return int((25000 <= totalsalary) and (totalsalary <= 75000))

    @staticmethod
    def _classification_function_6(salary, commission, age, elevel, car, zipcode, hvalue, hyears,
                                   loan):
        disposable = (2 * (salary + commission) / 3 - loan / 5 - 20000)
        return 0 if disposable > 1 else 1

    @staticmethod
    def _classification_function_7(salary, commission, age, elevel, car, zipcode, hvalue,
                                   hyears, loan):
        disposable = (2 * (salary + commission) / 3 - 5000 * elevel - 20000)
        return 0 if disposable > 1 else 1

    @staticmethod
    def _classification_function_8(salary, commission, age, elevel, car, zipcode, hvalue,
                                   hyears, loan):
        disposable = (2 * (salary + commission) / 3 - 5000 * elevel - loan / 5 - 10000)
        return 0 if disposable > 1 else 1

    @staticmethod
    def _classification_function_9(salary, commission, age, elevel, car, zipcode, hvalue,
                                   hyears, loan):
        equity = 0
        if hyears >= 20:
            equity = hvalue * (hyears - 20) / 10
        disposable = (2 * (salary + commission) / 3 - 5000 * elevel + equity / 5 - 10000)
        return 0 if disposable > 1 else 1

    @property
    def _repr_content(self):
        return {**super()._repr_content,
                'Function': str(self.classification_function)}
