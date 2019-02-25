import math
from collections import defaultdict

from . import base
from . import count
from . import categorical_count


class GiniIndex(base.RunningStatistic):
    """Computes a running entropy.

    Attributes:
        gini_index : The running entropy.
        alpha (int) : Fading factor

    Exemples : 

        ::


    References :
    1. `Updating Formulas and Algorithms for Computing Entropy and Gini Index from Time-Changing Data Streams <https://arxiv.org/pdf/1403.6348.pdf>`_

    """

    def __init__(self, alpha=1):

        if 0 < alpha <= 1:
            self.alpha = alpha
        else:
            raise ValueError('alpha must be between 0  excluded and 1')
        self.gini_index = 0
        self.count = count.Count()
        self.count_category = categorical_count.CategoricalCount()

    @property
    def name(self):
        return 'gini_index'

    def update(self, x):

        if x not in self.count_category.get():
            self.count_category.cat_count[x] = 0
        n = self.count.get()
        ni = self.count_category.get()[x]
        numerator = ( ( n**2 ) * ( 1 - ( self.alpha * self.gini_index ) ) ) + 2*ni + 1
        denominator = (n + 1) **2
        self.gini_index = 1 - (numerator / denominator)
        
        self.count.update()
        self.count_category.update(x)
        return self

    def get(self):
        return self.gini_index
