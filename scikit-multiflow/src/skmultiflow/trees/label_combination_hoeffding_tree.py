import numpy as np
from skmultiflow.core import MultiOutputMixin
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.utils import get_dimensions

from ._nodes import LCActiveLearningNodeMC
from ._nodes import LCInactiveLearningNodeMC
from ._nodes import LCActiveLearningNodeNB
from ._nodes import LCActiveLearningNodeNBA

import warnings


def LCHT(max_byte_size=33554432, memory_estimate_period=1000000, grace_period=200,
         split_criterion='info_gain', split_confidence=0.0000001, tie_threshold=0.05,
         binary_split=False, stop_mem_management=False, remove_poor_atts=False, no_preprune=False,
         leaf_prediction='nba', nb_threshold=0, nominal_attributes=None,
         n_labels=None):     # pragma: no cover
    warnings.warn("'LCHT' has been renamed to 'LabelCombinationHoeffdingTreeClassifier' in"
                  "v0.5.0.\nThe old name will be removed in v0.7.0", category=FutureWarning)
    return LabelCombinationHoeffdingTreeClassifier(max_byte_size=max_byte_size,
                                                   memory_estimate_period=memory_estimate_period,
                                                   grace_period=grace_period,
                                                   split_criterion=split_criterion,
                                                   split_confidence=split_confidence,
                                                   tie_threshold=tie_threshold,
                                                   binary_split=binary_split,
                                                   stop_mem_management=stop_mem_management,
                                                   remove_poor_atts=remove_poor_atts,
                                                   no_preprune=no_preprune,
                                                   leaf_prediction=leaf_prediction,
                                                   nb_threshold=nb_threshold,
                                                   nominal_attributes=nominal_attributes,
                                                   n_labels=n_labels)


class LabelCombinationHoeffdingTreeClassifier(HoeffdingTreeClassifier, MultiOutputMixin):
    """ Label Combination Hoeffding Tree for multi-label classification.

    Label combination transforms the problem from multi-label to multi-class.
    For each unique combination of labels it assigns a class and proceeds
    with training the hoeffding tree normally.

    The transformation is done by changing the label set which could be seen
    as a binary number to an int which will represent the class, and after
    the prediction the int is converted back to a binary number which is the
    predicted label-set.

    The number of labels need to be provided for the transformation to work.

    Parameters
    ----------
    max_byte_size: int (default=33554432)
        Maximum memory consumed by the tree.
    memory_estimate_period: int (default=1000000)
        Number of instances between memory consumption checks.
    grace_period: int (default=200)
        Number of instances a leaf should observe between split attempts.
    split_criterion: string (default='info_gain')
        | Split criterion to use.
        | 'gini' - Gini
        | 'info_gain' - Information Gain
    split_confidence: float (default=0.0000001)
        Allowed error in split decision, a value closer to 0 takes longer to decide.
    tie_threshold: float (default=0.05)
        Threshold below which a split will be forced to break ties.
    binary_split: boolean (default=False)
        If True, only allow binary splits.
    stop_mem_management: boolean (default=False)
        If True, stop growing as soon as memory limit is hit.
    remove_poor_atts: boolean (default=False)
        If True, disable poor attributes.
    no_preprune: boolean (default=False)
        If True, disable pre-pruning.
    leaf_prediction: string (default='nba')
        | Prediction mechanism used at leafs.
        | 'mc' - Majority Class
        | 'nb' - Naive Bayes
        | 'nba' - Naive Bayes Adaptive
    nb_threshold: int (default=0)
        Number of instances a leaf should observe before allowing Naive Bayes.
    nominal_attributes: list, optional
        List of Nominal attributes. If emtpy, then assume that all attributes are numerical.
    n_labels: int (default=None)
        the number of labels the problem has.

    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.data import MultilabelGenerator
    >>> from skmultiflow.trees import LabelCombinationHoeffdingTreeClassifier
    >>> from skmultiflow.metrics import hamming_score
    >>>
    >>> # Setting up a data stream
    >>> stream = MultilabelGenerator(random_state=1, n_samples=200,
    >>>                              n_targets=5, n_features=10)
    >>>
    >>> # Setup Label Combination Hoeffding Tree classifier
    >>> lc_ht = LabelCombinationHoeffdingTreeClassifier(n_labels=stream.n_targets)
    >>>
    >>> # Setup variables to control loop and track performance
    >>> n_samples = 0
    >>> max_samples = 200
    >>> true_labels = []
    >>> predicts = []
    >>>
    >>> # Train the estimator with the samples provided by the data stream
    >>> while n_samples < max_samples and stream.has_more_samples():
    >>>     X, y = stream.next_sample()
    >>>     y_pred = lc_ht.predict(X)
    >>>     lc_ht.partial_fit(X, y)
    >>>     predicts.extend(y_pred)
    >>>     true_labels.extend(y)
    >>>     n_samples += 1
    >>>
    >>> # Display results
    >>> perf = hamming_score(true_labels, predicts)
    >>> print('{} samples analyzed.'.format(n_samples))
    >>> print('Label Combination Hoeffding Tree Hamming score: ' + str(perf))
    """
    def __init__(self,
                 max_byte_size=33554432,
                 memory_estimate_period=1000000,
                 grace_period=200,
                 split_criterion='info_gain',
                 split_confidence=0.0000001,
                 tie_threshold=0.05,
                 binary_split=False,
                 stop_mem_management=False,
                 remove_poor_atts=False,
                 no_preprune=False,
                 leaf_prediction='nba',
                 nb_threshold=0,
                 nominal_attributes=None,
                 n_labels=None):

        super().__init__(max_byte_size=max_byte_size,
                         memory_estimate_period=memory_estimate_period,
                         grace_period=grace_period,
                         split_criterion=split_criterion,
                         split_confidence=split_confidence,
                         tie_threshold=tie_threshold,
                         binary_split=binary_split,
                         stop_mem_management=stop_mem_management,
                         remove_poor_atts=remove_poor_atts,
                         no_preprune=no_preprune,
                         leaf_prediction=leaf_prediction,
                         nb_threshold=nb_threshold,
                         nominal_attributes=nominal_attributes)
        self.n_labels = n_labels

    @property
    def n_labels(self):
        return self._n_labels

    @n_labels.setter
    def n_labels(self, n_labels):
        if n_labels is None:
            raise ValueError('The number of labels must be specified')
        self._n_labels = n_labels

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """ Incrementally trains the model. Train samples (instances) are composed of X attributes
        and their corresponding targets y.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Instance attributes.
        y: array_like
            Classes (targets) for all samples in X.
        classes: Not used (default=None)
        sample_weight: float or array-like, optional (default=None)
            Samples weight. If not provided, uniform weights are assumed.

        Returns
        -------
            self
            """
        super().partial_fit(X, y, sample_weight=sample_weight)    # Override HT, infer the classes

    def predict(self, X):
        """Predicts the label of the X instance(s)

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Samples for which we want to predict the labels.

        Returns
        -------
        numpy.array
            Predicted labels for all instances in X.

        """
        r, _ = get_dimensions(X)
        predictions = []
        y_proba = self.predict_proba(X)
        for i in range(r):
            index = np.argmax(y_proba[i])
            pred = str("{0:0"+str(self.n_labels)+"b}").format(index)
            pred = [int(e) for e in pred]
            predictions.append(pred)

        return np.array(predictions)

    def _new_learning_node(self, initial_class_observations=None, is_active=True):
        """Create a new learning node. The type of learning node depends on the tree
        configuration."""
        if initial_class_observations is None:
            initial_class_observations = {}

        if is_active:
            if self._leaf_prediction == self._MAJORITY_CLASS:
                return LCActiveLearningNodeMC(initial_class_observations)
            elif self._leaf_prediction == self._NAIVE_BAYES:
                return LCActiveLearningNodeNB(initial_class_observations)
            else:  # NAIVE BAYES ADAPTIVE (default)
                return LCActiveLearningNodeNBA(initial_class_observations)
        else:
            return LCInactiveLearningNodeMC(initial_class_observations)

    @staticmethod
    def _more_tags():
        return {'multioutput': True,
                'multioutput_only': True}
