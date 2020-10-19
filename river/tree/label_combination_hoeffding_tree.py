from collections import defaultdict

from river import base
from river.tree import HoeffdingTreeClassifier
from river.utils.math import softmax


class LabelCombinationHoeffdingTreeClassifier(HoeffdingTreeClassifier, base.MultiOutputMixin):
    """Label Combination Hoeffding Tree for multi-label classification.

    Label combination transforms the problem from multi-label to multi-class.
    For each unique combination of labels it assigns a class and proceeds
    with training the hoeffding tree normally.

    The transformation is done by changing the label set which could be seen
    as a binary number to an int which will represent the class, and after
    the prediction the int is converted back to a binary number which is the
    predicted label-set.

    Parameters
    ----------
    grace_period
        Number of instances a leaf should observe between split attempts.
    split_criterion
        | Split criterion to use.
        | 'gini' - Gini
        | 'info_gain' - Information Gain
        | 'hellinger' - Helinger Distance
    split_confidence
        Allowed error in split decision, a value closer to 0 takes longer to decide.
    tie_threshold
        Threshold below which a split will be forced to break ties.
    leaf_prediction
        | Prediction mechanism used at leafs.
        | 'mc' - Majority Class
        | 'nb' - Naive Bayes
        | 'nba' - Naive Bayes Adaptive
    nb_threshold
        Number of instances a leaf should observe before allowing Naive Bayes.
    nominal_attributes
        List of Nominal attributes identifiers. If empty, then assume that all numeric attributes
        should be treated as continuous.
    **kwargs
        Other parameters passed to river.tree.DecisionTree.

    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.data import MultilabelGenerator
    >>> from skmultiflow.trees import LabelCombinationHoeffdingTreeClassifier
    >>> from skmultiflow.metrics import hamming_score

    >>> # Setting up a data stream
    >>> stream = MultilabelGenerator(seed=1, n_samples=200,
    >>>                              n_targets=5, n_features=10)

    >>> # Setup Label Combination Hoeffding Tree classifier
    >>> lc_ht = LabelCombinationHoeffdingTreeClassifier(n_labels=stream.n_targets)

    >>> # Setup variables to control loop and track performance
    >>> n_samples = 0
    >>> max_samples = 200
    >>> true_labels = []
    >>> predicts = []

    >>> # Train the estimator with the samples provided by the data stream
    >>> while n_samples < max_samples and stream.has_more_samples():
    >>>     X, y = stream.next_sample()
    >>>     y_pred = lc_ht.predict(X)
    >>>     lc_ht.partial_fit(X, y)
    >>>     predicts.extend(y_pred)
    >>>     true_labels.extend(y)
    >>>     n_samples += 1

    >>> # Display results
    >>> perf = hamming_score(true_labels, predicts)
    >>> print('{} samples analyzed.'.format(n_samples))
    >>> print('Label Combination Hoeffding Tree Hamming score: ' + str(perf))
    """
    def __init__(self,
                 grace_period: int = 200,
                 split_criterion: str = 'info_gain',
                 split_confidence: float = 1e-7,
                 tie_threshold: float = 0.05,
                 leaf_prediction: str = 'nba',
                 nb_threshold: int = 0,
                 nominal_attributes: list = None,
                 **kwargs):

        super().__init__(grace_period=grace_period,
                         split_criterion=split_criterion,
                         split_confidence=split_confidence,
                         tie_threshold=tie_threshold,
                         leaf_prediction=leaf_prediction,
                         nb_threshold=nb_threshold,
                         nominal_attributes=nominal_attributes,
                         **kwargs)

        self._next_label_code = 0
        self._label_map = {}
        self._r_label_map = {}

    def reset(self):
        super().reset()

        self._next_label_code = 0
        self._label_map = {}
        self._r_label_map = {}

        return self

    def learn_one(self, x, y, *, sample_weight=1.):
        """ Update the Multi-label Hoeffding Tree Classifier.

        Parameters
        ----------
        x
            Instance attributes.
        y
            Labels of the instance.
        sample_weight
            The weight of the sample.

        Returns
        -------
            self
        """
        aux_label = tuple(sorted(y.items()))
        if aux_label not in self._label_map:
            self._label_map[aux_label] = self._next_label_code
            self._r_label_map[self._next_label_code] = aux_label
            self._next_label_code += 1
        y_encoded = self._label_map[aux_label]

        super().learn_one(x, y_encoded, sample_weight=sample_weight)

        return self

    def predict_proba_one(self, x):
        if self._tree_root is None:
            return None

        class_probas = super().predict_proba_one(x)

        labels_proba = defaultdict(lambda: {0: 0., 1: 0.})

        # Assign class probas to each label
        for code, proba in class_probas.items():
            for label_id, label_val in self._r_label_map[code]:
                aux = labels_proba[label_id]
                aux[label_val] += proba
                labels_proba[label_id] = aux

        # Normalize the data
        result = {}
        for label_id in labels_proba:
            result[label_id] = softmax(labels_proba[label_id])

        return result

    def predict_one(self, x):
        """Predict the labels of an instance.

        Parameters
        ----------
        x
            The instance for which we want to predict labels.

        Returns
        -------
            Predicted labels.

        """
        if self._tree_root is None:
            return None

        probas = self.predict_proba_one(x)

        preds = {}
        for label_id, label_probas in probas.items():
            preds[label_id] = max(label_probas, key=label_probas.get)

        return preds
