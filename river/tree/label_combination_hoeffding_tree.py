import typing

from river import base
from river.tree import HoeffdingTreeClassifier
from river.utils.skmultiflow_utils import normalize_values_in_dict


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
    attr_obs
        The Attribute Observer (AO) used to monitor the class statistics of numeric
        features and perform splits. Parameters can be passed to the AOs (when supported)
        by using `attr_obs_params`. Valid options are:</br>
        - `'bst'`: Binary Search Tree.</br>
        - `'gaussian'`: Gaussian observer. The `n_splits` used to query
         for split candidates can be adjusted (defaults to `10`).</br>
        - `'histogram'`: Histogram-based class frequency estimation.  The number of histogram
        bins (`n_bins` -- defaults to `256`) and the number of split point candidates to
        evaluate (`n_splits` -- defaults to `32`) can be adjusted.</br>
        See 'Notes' for more information about the AOs.
    attr_obs_params
        Parameters passed to the numeric AOs. See `attr_obs` for more information.
    kwargs
        Other parameters passed to `river.tree.BaseHoeffdingTree`.

    Notes
    -----
    Hoeffding trees rely on Attribute Observer (AO) algorithms to monitor input features
    and perform splits. Nominal features can be easily dealt with, since the partitions
    are well-defined. Numerical features, however, require more sophisticated solutions.
    Currently, three AOs are supported in `river` for classification trees:

    - *Binary Search Tree (BST)*: uses an exhaustive algorithm to find split candidates,
    similarly to batch decision trees. It ends up storing all observations between split
    attempts. This AO is the most costly one in terms of memory and processing
    time; however, it tends to yield the most accurate results when using `leaf_prediction=mc`.
    It cannot be used to calculate the Probability Density Function (PDF) of the monitored
    feature due to its binary tree nature. Hence, leaf prediction strategies other than
    the majority class will end up effectively mimicing the majority class classifier.
    This AO has no parameters.</br>
    - *Gaussian Estimator*: Approximates the numeric feature distribution by using
    a Gaussian distribution per class. The Cumulative Distribution Function (CDF) necessary to
    calculate the entropy (and, consequently, the information gain), the gini index, and
    other split criteria is then calculated using the fit feature's distribution.</br>
    - *Histogram*: approximates the numeric feature distribution using an incrementally
    maintained histogram per class. It represents a compromise between the intensive
    resource usage of BST and the strong assumptions about the feature's distribution
    used in the Gaussian Estimator. Besides that, this AO sits in the middle between the
    previous two in terms of memory usage and running time. Note that the number of
    bins affects the probability density approximation required to use leaves with
    (adaptive) naive bayes models. Hence, Histogram tends to be less accurate than the
    Gaussian estimator when adaptive or naive bayes leaves are used.

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
        attr_obs: str = "gaussian",
        attr_obs_params: dict = None,
        **kwargs
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
            attr_obs=attr_obs,
            attr_obs_params=attr_obs_params,
            **kwargs
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
        self._labels.update(y.keys())

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
        if self._tree_root is None:
            return None

        probas = self.predict_proba_one(x)

        preds = {}
        for label_id, label_probas in probas.items():
            preds[label_id] = max(label_probas, key=label_probas.get)

        return preds
