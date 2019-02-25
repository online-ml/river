import math
from collections import defaultdict

from . import base
from . import count
from . import categorical_count


class Entropy(base.RunningStatistic):
    """Computes a running entropy.

    Attributes:
        entropy : The running entropy.
        alpha (int) : Fading factor
    
    Exemples : 

        ::

            >>> import math
            >>> import random
            >>> import numpy as np
            >>> from scipy.stats import entropy
            >>> from creme import stats
            
            >>> def entropy_list(labels, base=None):
            ...   value,counts = np.unique(labels, return_counts=True)
            ...   return entropy(counts, base=base)

            >>> SEED = 42 * 1337
            >>> random.seed(SEED)

            >>> entro = stats.Entropy(alpha=1) 

            >>> list_animal = []
            >>> for animal, num_val in zip(['cat', 'dog', 'bird'],[301, 401, 601]): 
            ...     list_animal += [animal for i in range(num_val)]
            >>> random.shuffle(list_animal)

            >>> for animal in list_animal:
            ...     _ = entro.update(animal)

            >>> print(f'{entro.get():.6f}')
            1.058093
            >>> print(f'{entropy_list(list_animal):.6f}')
            1.058093
       

    References :
    1. `Updating Formulas and Algorithms for Computing Entropy and Gini Index from Time-Changing Data Streams <https://arxiv.org/pdf/1403.6348.pdf>`_

    """

    def __init__(self, alpha=1, eps=1e-8):

        if 0 < alpha <= 1:
            self.alpha = alpha
        else:
            raise ValueError('alpha must be between 0  excluded and 1')
        self.eps = eps
        self.entropy = 0
        self.count = count.Count()
        self.count_category = categorical_count.CategoricalCount()

    @property
    def name(self):
        return 'entropy'

    def update(self, x):
        
        if x not in self.count_category.get():
            self.count_category.cat_count[x] = 0

        updated_entropy = ((self.count.get() + self.eps) / (self.count.get() + 1)) * ((self.alpha *
                                                                         self.entropy) - math.log(((self.count.get() + self.eps) / (self.count.get() + 1))))
        
        updated_entropy -= ((self.count_category.get()[x] + 1) / (self.count.get() + 1)) * math.log(
            ((self.count_category.get()[x] + 1) / (self.count.get() + 1)))
        
        updated_entropy += ((self.count_category.get()[x] + self.eps) / (self.count.get() + 1)) * math.log(
            ((self.count_category.get()[x] + self.eps) / (self.count.get() + 1)))
       
        self.entropy = updated_entropy

        self.count.update()
        self.count_category.update(x)
        return self

    def get(self):
        return self.entropy
