import math

from river import base

from .base import ActiveLearningClassifier


class FixedUncertainty(ActiveLearningClassifier):

    """Strategy of Active Learning to select instances more significative based onde uncertainty.
    
    Version 1.0
    
    Reference
    I. Zliobaite, A. Bifet, B.Pfahringer, G. Holmes. “Active Learning with Drifting Streaming Data”, IEEE Transactions on Neural Netowrks and Learning Systems, Vol.25 (1), pp.27-39, 2014.
 
    Parameters
    ----------
    classifier
        The classifier to wrap.
    maximum_posteriori
        The maximum posterior probability of a classifier. 
        More information in the paper of reference.
    theta
        Threshold of uncertainty. If posteriori is less to theta. So, the instance is selected for labeling.
        Default value: 0.95.
        More information in the paper of reference.
    discount_factor
        The discount factor to apply to the entropy measure. A value of 1 won't affect the entropy.
        The higher the discount factor, the more the entropy will be discounted, and the less
        likely samples will be selected for labeling. A value of 0 will select all samples for
        labeling. The discount factor is thus a way to control how many samples are selected for
        labeling.
    seed
        Random number generator seed for reproducibility.

    Examples (TO DO)
    --------


    """
    def __init__(self, classifier: base.Classifier, maximum_posteriori: float, theta: float = 0.95, seed=None):
        super().__init__(classifier, seed=seed)
        self.maximum_posteriori = maximum_posteriori
        self.theta = theta
    
    def _ask_for_label(self) -> bool:
        '''Version 1.0'''
        selected = False
        if self.maximum_posteriori < self.theta:
            selected = True
        return selected
