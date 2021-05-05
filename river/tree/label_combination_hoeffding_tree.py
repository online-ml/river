import typing
import warnings

from river import base, tree
from river.utils.skmultiflow_utils import normalize_values_in_dict

from .splitter import Splitter


class LabelCombinationHoeffdingTreeClassifier(
    tree.HoeffdingTreeClassifier, base.MultiOutputMixin
):
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
    max_depth
        The maximum depth a tree can reach. If `None`, the tree will grow indefinitely.
    split_criterion
        Split criterion to use.</br>
        - 'gini' - Gini</br>
        - 'info_gain' - Information Gain</br>
        - 'hellinger' - Helinger Distance</br>
    split_confidence
        Allowed error in split decision, a value closer to 0 takes longer to decide.
    tie_threshold
        Threshold below which a split will be forced to break ties.
    leaf_prediction
        Prediction mechanism used at leafs.</br>
        - 'mc' - Majority Class</br>
        - 'nb' - Naive Bayes</br>
        - 'nba' - Naive Bayes Adaptive</br>
    nb_threshold
        Number of instances a leaf should observe before allowing Naive Bayes.
    nominal_attributes
        List of Nominal attributes identifiers. If empty, then assume that all numeric attributes
        should be treated as continuous.
    splitter
        The Splitter or Attribute Observer (AO) used to monitor the class statistics of numeric
        features and perform splits. Splitters are available in the `tree.splitter` module.
        Different splitters are available for classification and regression tasks. Classification
        and regression splitters can be distinguished by their property `is_target_class`.
        This is an advanced option. Special care must be taken when choosing different splitters.
        By default, `tree.splitter.GaussianSplitter` is used if `splitter` is `None`.
    binary_split
        If True, only allow binary splits.
    max_size
        The max size of the tree, in Megabytes (MB).
    memory_estimate_period
        Interval (number of processed instances) between memory consumption checks.
    stop_mem_management
        If True, stop growing as soon as memory limit is hit.
    remove_poor_attrs
        If True, disable poor attributes to reduce memory usage.
    merit_preprune
        If True, enable merit-based tree pre-pruning.

    Examples
    --------
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import tree

    >>> dataset = iter(datasets.Music().take(200))
    >>> model = tree.LabelCombinationHoeffdingTreeClassifier(
    ...     split_confidence=1e-5,
    ...     grace_period=50
    ... )

    >>> metric = metrics.Hamming()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    Hamming: 0.154104
    """

    def __init__(
        self,
        grace_period: int = 200,
        max_depth: int = None,
        split_criterion: str = "info_gain",
        split_confidence: float = 1e-7,
        tie_threshold: float = 0.05,
        leaf_prediction: str = "nba",
        nb_threshold: int = 0,
        nominal_attributes: list = None,
        splitter: Splitter = None,
        binary_split: bool = False,
        max_size: int = 100,
        memory_estimate_period: int = 1000000,
        stop_mem_management: bool = False,
        remove_poor_attrs: bool = False,
        merit_preprune: bool = True,
    ):

        super().__init__(
            grace_period=grace_period,
            max_depth=max_depth,
            split_criterion=split_criterion,
            split_confidence=split_confidence,
            tie_threshold=tie_threshold,
            leaf_prediction=leaf_prediction,
            nb_threshold=nb_threshold,
            nominal_attributes=nominal_attributes,
            splitter=splitter,
            binary_split=binary_split,
            max_size=max_size,
            memory_estimate_period=memory_estimate_period,
            stop_mem_management=stop_mem_management,
            remove_poor_attrs=remove_poor_attrs,
            merit_preprune=merit_preprune,
        )

        self._next_label_code: int = 0
        self._label_map: typing.Dict[typing.Tuple, int] = {}
        self._r_label_map: typing.Dict[int, typing.Tuple] = {}
        self._labels: typing.Set[typing.Hashable] = set()

    def learn_one(self, x, y, *, sample_weight=1.0):
        """Update the Multi-label Hoeffding Tree Classifier.

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
        self._labels.update(y.keys())  # noqa

        aux_label = tuple(sorted(y.items()))  # noqa
        if aux_label not in self._label_map:
            self._label_map[aux_label] = self._next_label_code
            self._r_label_map[self._next_label_code] = aux_label
            self._next_label_code += 1
        y_encoded = self._label_map[aux_label]

        super().learn_one(x, y_encoded, sample_weight=sample_weight)

        return self

    def predict_proba_one(self, x):
        if self._root is None:
            return None

        enc_probas = super().predict_proba_one(x)
        enc_class = max(enc_probas, key=enc_probas.get)

        result = {}
        for lbl in self._labels:
            result[lbl] = {False: 0.0, True: 0.0}

        for label_id, label_val in self._r_label_map[enc_class]:
            result[label_id][label_val] = enc_probas[enc_class]
            result[label_id] = normalize_values_in_dict(result[label_id])

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
        if self._root is None:
            return None

        probas = self.predict_proba_one(x)

        preds = {}
        for label_id, label_probas in probas.items():
            preds[label_id] = max(label_probas, key=label_probas.get)

        return preds

    def debug_one(self, x: dict):
        warnings.warn(f"'debug_one' is not supported by {self.__class__.__name__}")

    def draw(self, max_depth: int = None):
        warnings.warn(f"'draw' is not supported by {self.__class__.__name__}")
