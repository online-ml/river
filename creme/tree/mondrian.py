"""

- [Decision Trees and Forests: A Probabilistic Perspective](http://www.gatsby.ucl.ac.uk/~balaji/balaji-phd-thesis.pdf)
- [Mondrian Forests for Large-Scale Regression when Uncertainty Matters](http://www.gatsby.ucl.ac.uk/~balaji/mfregression_aistats16.pdf)
- [Mondrian Forests: Efficient Online Random Forests](https://papers.nips.cc/paper/5234-mondrian-forests-efficient-online-random-forests.pdf)
- [Intuition behind Mondrian Trees](https://scikit-garden.github.io/examples/MondrianTreeRegressor/)
- [Mondrian Forest](https://ldocao.wordpress.com/2016/08/26/mondrian-forest/)
"""
import collections
import copy
import math
import random

import numpy as np
from sklearn import utils


Split = collections.namedtuple('Split', 'feature threshold')

Bound = collections.namedtuple('Bound', 'lower upper')


class Block(collections.defaultdict):

    def __init__(self, bounds=None):
        super().__init__(lambda: Bound(0., 0.))
        for feature, bound in (bounds or {}).items():
            self[feature] = bound

    def contains(self, x):
        """
        Examples
        --------
        >>> block = Block({'x': Bound(-1, 1)})
        >>> block.contains({'x': 0.5})
        True
        >>> block.contains({'x': -2})
        False
        >>> block.contains({'x': 2})
        False
        """
        if self:
            return all(bound.lower <= x[feature] <= bound.upper for feature, bound in self.items())
        return True


class MondrianNode:

    def __init__(self, tau, feature, threshold, left=None, right=None):
        self.tau = tau
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

    @property
    def is_leaf(self):
        return self.left is None and self.right is None

    def extend(self, x):
        e_l = {i: max(self.block[i].lower - xi, 0) for i, xi in x.items()}
        e_u = {i: max(xi - self.block[i].upper, 0) for i, xi in x.items()}

        #
        rate = sum(e_l.values()) + sum(e_u.values())
        E = np.random.exponential(scale=1 / rate, size=1)[0]

        self.block = new_block
        return self

    def predict(self, x):

        if x[self.feature] < self.threshold:
            return self.left.predict(x)
        return self.right.predict(x)


        #rate = sum(u - l for u, l in zip(self.lower_bounds, self.upper_bounds))
        #    time = np.random.exponential(scale=1 / rate, size=1)[0]


class MondrianTree:

    def __init__(self, lifetime, random_state=None):
        self.lifetime = lifetime
        self.lower_bounds = collections.defaultdict(float)
        self.upper_bounds = collections.defaultdict(float)
        self.rng = utils.check_random_state(random_state)

    def _sample_mondrian_block(self, x):
        rate = 2 * sum(x.values())
        E = np.random.exponential(scale=1 / rate, size=1)[0]
        if E < self.lifetime:
            feature = random.choice(features)
            return MondrianNode(tau=E, feature=feature, threshold=x[feature])
        return MondrianNode(tau=self.lifetime, feature=None, threshold=None)

    #def fit(self, x, y):
    #    self.tree_ = self._sample_mondrian_block()

