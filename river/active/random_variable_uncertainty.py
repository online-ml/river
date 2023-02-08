from river import base

import numpy as np

from .base import ActiveLearningClassifier


class RandomVariableUncertainty (ActiveLearningClassifier):
    
    

    """Strategy of Active Learning to select instances more significative based on uncertainty.

    Version 1.0.

    Reference
    I. Zliobaite, A. Bifet, B.Pfahringer, G. Holmes. “Active Learning with Drifting Streaming Data”, IEEE Transactions on Neural Netowrks and Learning Systems, Vol.25 (1), pp.27-39, 2014.

    Parameters
    ----------
    classifier
        The classifier to wrap.
    theta
        Threshold of uncertainty. If posteriori is less to theta. So, the instance is selected for labeling.
        Default value: 0.95.
        More information in the paper of reference.
    s
        Threshold adjustment step $$s \in (0,1].$$
        Default value: 0.5.
        More information in the paper of reference.
                                           
                                           
    delta
        Variance of the threshold randomization.
        Default value: 1.00.
        More information in the paper of reference.
        
    seed
        Random number generator seed for reproducibility.

    >>> from river import active
    >>> from river import datasets
    >>> from river import feature_extraction
    >>> from river import linear_model

    >>> dataset = datasets.SMSSpam()

    >>> model = (
            feature_extraction.TFIDF(on='body') |
            linear_model.LogisticRegression()
         )
    >>> model = active.RandomVariableUncertainty(model, seed=42)

    >>> for x, y in dataset:

            ## IF ask = True
            ##    Selected instance.
            ## IF ask = False
            ##    Instance not selected
            ## The FixedUncertainty use the maximium
            ## posterior probability. So use only the predict_proba_one(X).
            ## Do not use predict_one(x).
            >>> y_pred, ask = model.predict_proba_one(x)


    --------


    """
    
    

    def __init__(self, classifier: base.Classifier, theta: float = 0.95, s=0.5, delta=1.0, seed=None):
        super().__init__(classifier, seed=seed)
      
        self.theta = theta
        self.s = s
        self.delta = delta
        
        
    @property
    def s(self):
        return self.s
    
    @s.setter
    def s(self,value):
        self._s = value
        
        
    @property
    def delta(self):
        return self.delta
    
    @delta.setter
    def s(self,value):
        self._delta = value
    
    
    @property
    def theta(self):
        return self.theta
    
    @theta.setter
    def s(self,value):
        self._theta = value

    def _ask_for_label(self, x, y_pred) -> bool:
        """Version 1.0"""
        maximum_posteriori = max(y_pred.values())
        selected = False
        
        thetaRand = self.theta * np.random.normal(1,self.delta)

        if maximum_posteriori < thetaRand:
            self.theta = self.theta*(1-self.s)
            selected = True
        else:
            self.theta = self.theta*(1+self.s)
            selected = False
            
        return selected
