from __future__ import annotations

import random

from river import datasets


class Agrawal(datasets.base.SyntheticDataset):
    r"""Agrawal stream generator.

    The generator was introduced by Agrawal et al. [^1], and was a common
    source of data for early work on scaling up decision tree learners.
    The generator produces a stream containing nine features, six numeric and
    three categorical.
    There are 10 functions defined for generating binary class labels from the
    features. Presumably these determine whether the loan should be approved.
    Classification functions are listed in the original paper [^1].

    **Feature** | **Description** | **Values**

    * `salary` | salary | uniformly distributed from 20k to 150k

    * `commission` | commission | 0 if `salary` < 75k else uniformly distributed from 10k to 75k

    * `age` | age | uniformly distributed from 20 to 80

    * `elevel` | education level | uniformly chosen from 0 to 4

    * `car` | car maker | uniformly chosen from 1 to 20

    * `zipcode` | zip code of the town | uniformly chosen from 0 to 8

    * `hvalue` | house value | uniformly distributed from 50k x zipcode to 100k x zipcode

    * `hyears` | years house owned | uniformly distributed from 1 to 30

    * `loan` | total loan amount | uniformly distributed from 0 to 500k

    Parameters
    ----------
    classification_function
        The classification function to use for the generation.
        Valid values are from 0 to 9.
    seed
        Random seed for reproducibility.
    balance_classes
        If True, the class distribution will converge to a uniform distribution.
    perturbation
        The probability that noise will happen in the generation. Each new
        sample will be perturbed by the magnitude of `perturbation`.
        Valid values are in the range [0.0 to 1.0].

    Examples
    --------

    >>> from river.datasets import synth

    >>> dataset = synth.Agrawal(
    ...     classification_function=0,
    ...     seed=42
    ... )

    >>> dataset
    Synthetic data generator
    <BLANKLINE>
        Name  Agrawal
        Task  Binary classification
     Samples  ∞
    Features  9
     Outputs  1
     Classes  2
      Sparse  False
    <BLANKLINE>
    Configuration
    -------------
    classification_function  0
                       seed  42
            balance_classes  False
               perturbation  0.0

    >>> for x, y in dataset.take(5):
    ...     print(list(x.values()), y)
    [103125.4837, 0, 21, 2, 8, 3, 319768.9642, 4, 338349.7437] 1
    [135983.3438, 0, 25, 4, 14, 0, 423837.7755, 7, 116330.4466] 1
    [98262.4347, 0, 55, 1, 18, 6, 144088.1244, 19, 139095.3541] 0
    [133009.0417, 0, 68, 1, 14, 5, 233361.4025, 7, 478606.5361] 1
    [63757.2908, 16955.9382, 26, 2, 12, 4, 522851.3093, 24, 229712.4398] 1

    Notes
    -----
    The sample generation works as follows: The 9 features are generated
    with the random generator, initialized with the seed passed by the
    user. Then, the classification function decides, as a function of all
    the attributes, whether to classify the instance as class 0 or class
    1. The next step is to verify if the classes should be balanced, and
    if so, balance the classes. Finally, add noise if `perturbation` > 0.0.

    References
    ----------
    [^1]: Rakesh Agrawal, Tomasz Imielinksi, and Arun Swami. "Database Mining:
          A Performance Perspective", IEEE Transactions on Knowledge and
          Data Engineering, 5(6), December 1993.

    """

    def __init__(
        self,
        classification_function: int = 0,
        seed: int | None = None,
        balance_classes: bool = False,
        perturbation: float = 0.0,
    ):
        super().__init__(n_features=9, n_classes=2, n_outputs=1, task=datasets.base.BINARY_CLF)

        # Classification functions to use
        self._classification_functions = [
            self._classification_function_0,
            self._classification_function_1,
            self._classification_function_2,
            self._classification_function_3,
            self._classification_function_4,
            self._classification_function_5,
            self._classification_function_6,
            self._classification_function_7,
            self._classification_function_8,
            self._classification_function_9,
        ]
        if classification_function not in range(10):
            raise ValueError(
                f"classification_function takes values from 0 to 9 "
                f"and {classification_function} was passed"
            )
        self.classification_function = classification_function
        self.balance_classes = balance_classes
        if not 0.0 <= perturbation <= 1.0:
            raise ValueError(
                f"noise percentage should be in [0.0..1.0] " f"and {perturbation} was passed"
            )
        self.perturbation = perturbation
        self.seed = seed
        self.n_num_features = 6
        self.n_cat_features = 3
        self._next_class_should_be_zero = False
        self.feature_names = [
            "salary",
            "commission",
            "age",
            "elevel",
            "car",
            "zipcode",
            "hvalue",
            "hyears",
            "loan",
        ]
        self.target_values = [i for i in range(self.n_classes)]

    def __iter__(self):
        self._rng = random.Random(self.seed)
        self._next_class_should_be_zero = False

        while True:
            y = 0
            desired_class_found = False
            while not desired_class_found:
                salary = 20000 + 130000 * self._rng.random()
                commission = 0 if (salary >= 75000) else (10000 + 75000 * self._rng.random())
                age = self._rng.randint(20, 80)
                elevel = self._rng.randint(0, 4)
                car = self._rng.randint(1, 20)
                zipcode = self._rng.randint(0, 8)
                hvalue = (8 - zipcode) * 100000 * (0.5 + self._rng.random())
                hyears = self._rng.randint(1, 30)
                loan = self._rng.random() * 500000
                y = self._classification_functions[self.classification_function](
                    salary, commission, age, elevel, car, zipcode, hvalue, hyears, loan
                )
                if not self.balance_classes:
                    desired_class_found = True
                else:
                    if (self._next_class_should_be_zero and (y == 0)) or (
                        (not self._next_class_should_be_zero) and (y == 1)
                    ):
                        desired_class_found = True
                        self._next_class_should_be_zero = not self._next_class_should_be_zero

            if self.perturbation > 0.0:
                salary = self._perturb_value(salary, 20000, 150000)
                if commission > 0:
                    commission = self._perturb_value(commission, 10000, 75000)
                age = round(self._perturb_value(age, 20, 80))
                hvalue = self._perturb_value(hvalue, (9 - zipcode) * 100000, 0, 135000)
                hyears = round(self._perturb_value(hyears, 1, 30))
                loan = self._perturb_value(loan, 0, 500000)

            x = dict()
            for feature in self.feature_names:
                x[feature] = eval(feature)

            yield x, y

    def _perturb_value(self, val, val_min, val_max, val_range=None):
        if val_range is None:
            val_range = val_max - val_min
        val += val_range * (2 * (self._rng.random() - 0.5)) * self.perturbation
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
    def _classification_function_0(
        salary, commission, age, elevel, car, zipcode, hvalue, hyears, loan
    ):
        return int((age < 40) or (60 <= age))

    @staticmethod
    def _classification_function_1(
        salary, commission, age, elevel, car, zipcode, hvalue, hyears, loan
    ):
        if age < 40:
            return int((50000 <= salary) and (salary <= 100000))
        elif age < 60:
            return int((75000 <= salary) and (salary <= 125000))
        else:
            return int((25000 <= salary) and (salary <= 75000))

    @staticmethod
    def _classification_function_2(
        salary, commission, age, elevel, car, zipcode, hvalue, hyears, loan
    ):
        if age < 40:
            return int((elevel == 0) or (elevel == 1))
        elif age < 60:
            return int((elevel == 1) or (elevel == 2) or (elevel == 3))
        else:
            return int((elevel == 2) or (elevel == 3) or (elevel == 4))

    @staticmethod
    def _classification_function_3(
        salary, commission, age, elevel, car, zipcode, hvalue, hyears, loan
    ):
        if age < 40:
            if (elevel == 0) or (elevel == 1):
                return int(25000 <= salary <= 75000)
            else:
                return int(50000 <= salary <= 100000)
        elif age < 60:
            if (elevel == 1) or (elevel == 2) or (elevel == 3):
                return int(50000 <= salary <= 100000)
            else:
                return int(75000 <= salary <= 125000)
        else:
            if (elevel == 2) or (elevel == 3) or (elevel == 4):
                return int(50000 <= salary <= 100000)
            else:
                return int(25000 <= salary <= 75000)

    @staticmethod
    def _classification_function_4(
        salary, commission, age, elevel, car, zipcode, hvalue, hyears, loan
    ):
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
    def _classification_function_5(
        salary, commission, age, elevel, car, zipcode, hvalue, hyears, loan
    ):
        totalsalary = salary + commission

        if age < 40:
            return int((50000 <= totalsalary) and (totalsalary <= 100000))
        elif age < 60:
            return int((75000 <= totalsalary) and (totalsalary <= 125000))
        else:
            return int((25000 <= totalsalary) and (totalsalary <= 75000))

    @staticmethod
    def _classification_function_6(
        salary, commission, age, elevel, car, zipcode, hvalue, hyears, loan
    ):
        disposable = 2 * (salary + commission) / 3 - loan / 5 - 20000
        return 0 if disposable > 1 else 1

    @staticmethod
    def _classification_function_7(
        salary, commission, age, elevel, car, zipcode, hvalue, hyears, loan
    ):
        disposable = 2 * (salary + commission) / 3 - 5000 * elevel - 20000
        return 0 if disposable > 1 else 1

    @staticmethod
    def _classification_function_8(
        salary, commission, age, elevel, car, zipcode, hvalue, hyears, loan
    ):
        disposable = 2 * (salary + commission) / 3 - 5000 * elevel - loan / 5 - 10000
        return 0 if disposable > 1 else 1

    @staticmethod
    def _classification_function_9(
        salary, commission, age, elevel, car, zipcode, hvalue, hyears, loan
    ):
        equity = 0
        if hyears >= 20:
            equity = hvalue * (hyears - 20) / 10
        disposable = 2 * (salary + commission) / 3 - 5000 * elevel + equity / 5 - 10000
        return 0 if disposable > 1 else 1
