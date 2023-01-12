from river import base

import random


class Uncertainty(): 
    """
        Strategy of Active Learning to select instances more significative.
        Version 1.0
        
        Reference
        I. Zliobaite, A. Bifet, B.Pfahringer, G. Holmes. “Active Learning with Drifting Streaming Data”, IEEE Transactions on Neural Netowrks and Learning Systems, Vol.25 (1), pp.27-39, 2014.
    """

    def fixed_uncertainty(self,maximum_posteriori, theta):
         
        selected = False
        if maximum_posteriori < theta:
            selected = True
        return selected

    def variable_uncertainty(self, maximum_posteriori, s, theta):
        selected = False
        if maximum_posteriori < theta:
            theta = theta*(1-s)
            selected = True
        else:
            theta = theta*(1+s)
            selected = False
        return selected
        
    def random_variable_uncertainty(self, maximum_posteriori, s, delta, theta):
        selected = False
        n = random.gauss(1, delta)
        thetaRand = theta*n
        if maximum_posteriori < thetaRand:
            theta = theta*(1-s)
            selected = True
        else:
            theta = theta*(1+s)
            selected = False

        return selected


"""Example function with types documented in the docstring.

        `PEP 484`_ type annotations are supported. If attribute, parameter, and
        return types are annotated according to `PEP 484`_, they do not need to be
        included in the docstring:

        Args:
            param1 (int): The first parameter.
            param2 (str): The second parameter.

        Returns:
            bool: The return value. True for success, False otherwise.

        .. _PEP 484:
            https://www.python.org/dev/peps/pep-0484/

        """