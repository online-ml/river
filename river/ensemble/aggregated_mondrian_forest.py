import abc
import random

from river import base
from river.tree.mondrian import MondrianTreeClassifier


class AMFLearner(base.Ensemble, abc.ABC):
    """Base class for Aggregated Mondrian Forest classifier and regressors for online learning.

    Parameters
    ----------
    n_estimators
        The number of trees in the forest.
    step
         Step-size for the aggregation weights.
    loss
        The loss used for the computation of the aggregation weights.
    use_aggregation
        Controls if aggregation is used in the trees. It is highly recommended to
        leave it as `True`.
    split_pure
        Controls if nodes that contains only sample of the same class should be
        split ("pure" nodes). Default is `False`, namely pure nodes are not split,
        but `True` can be sometimes better.
    seed
        Random seed for reproducibility.

    Note
    ----
    This class is not intended for end users but for development only.
    """

    def __init__(
        self,
        n_estimators: int = 10,
        step: float = 0.1,
        loss: str = "log",
        use_aggregation: bool = True,
        split_pure: bool = False,
        seed: int = None,
    ):
        super().__init__([])  # type: ignore

        self.n_estimators = n_estimators
        self.step = step
        self.loss = loss
        self.use_aggregation = use_aggregation
        self.split_pure = split_pure
        self.seed = seed

        self._rng = random.Random(self.seed)

    def is_trained(self) -> bool:
        """Indicate whether the model has been trained at least once before."""
        return len(self) > 0

    @abc.abstractmethod
    def _initialize_trees(self):
        """Initialize the forest members."""

    @property
    def _min_number_of_models(self):
        return 0


class AMFClassifier(AMFLearner, base.Classifier):
    """Aggregated Mondrian Forest classifier for online learning. This algorithm
    is truly online, in the sense that a single pass is performed, and that predictions
    can be produced anytime.

    Each node in a tree predicts according to the distribution of the labels
    it contains. This distribution is regularized using a "Jeffreys" prior
    with parameter `dirichlet`. For each class with `count` labels in the
    node and `n_samples` samples in it, the prediction of a node is given by

    $\\frac{count + dirichlet}{n_{samples} + dirichlet \\times n_{classes}}$.

    The prediction for a sample is computed as the aggregated predictions of all the
    subtrees along the path leading to the leaf node containing the sample. The
    aggregation weights are exponential weights with learning rate `step` and log-loss
    when `use_aggregation` is `True`.

    This computation is performed exactly thanks to a context tree weighting algorithm.
    More details can be found in the paper cited in the references below.

    The final predictions are the average class probabilities predicted by each of the
    `n_estimators` trees in the forest.

    Parameters
    ----------
    n_classes
        Number of expected classes in the labels. This is required since we
        don't know the number of classes in advance in a online setting.
    n_estimators
        The number of trees in the forest.
    step
        Step-size for the aggregation weights. Default is 1 for classification with
        the log-loss, which is usually the best choice.
    use_aggregation
        Controls if aggregation is used in the trees. It is highly recommended to
        leave it as `True`.
    dirichlet
        Regularization level of the class frequencies used for predictions in each
        node. Default is dirichlet=0.5 for binary problems and dirichlet=0.01 otherwise.
    split_pure
        Controls if nodes that contains only sample of the same class should be
        split ("pure" nodes). Default is `False`, namely pure nodes are not split,
        but `True` can be sometimes better.
    seed
        Random seed for reproducibility.

    Notes
    -----
    Only log_loss used for the computation of the aggregation weights is supported for now, namely the log-loss
    for multi-class classification.

    References
    ----------
    J. Mourtada, S. Gaiffas and E. Scornet, *AMF: Aggregated Mondrian Forests for Online Learning*, arXiv:1906.10529, 2019.

    """

    def __init__(
        self,
        n_classes: int = 2,
        n_estimators: int = 10,
        step: float = 1.0,
        use_aggregation: bool = True,
        dirichlet: float = None,
        split_pure: bool = False,
        seed: int = None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            step=step,
            loss="log",
            use_aggregation=use_aggregation,
            split_pure=split_pure,
            seed=seed,
        )

        self.n_classes = n_classes
        if dirichlet is None:
            if self.n_classes == 2:
                self.dirichlet = 0.5
            else:
                self.dirichlet = 0.01
        else:
            self.dirichlet = dirichlet

        # memory of the classes
        self._classes: set[base.typing.ClfTarget] = set()
        self.iteration = 0

    def _initialize_trees(self):
        self.data: list[MondrianTreeClassifier] = []
        for _ in range(self.n_estimators):
            # We don't want to have the same stochastic scheme for each tree, or it'll break the randomness
            # Hence we introduce a new seed for each, that is derived of the given seed by a deterministic process
            seed = self._rng.randint(0, 9999999)

            tree = MondrianTreeClassifier(
                self.n_classes,
                self.step,
                self.use_aggregation,
                self.dirichlet,
                self.split_pure,
                self.iteration,
                seed,
            )
            self.data.append(tree)

    def learn_one(self, x, y):
        # Updating the previously seen classes with the new sample
        self._classes.add(y)

        # Checking if the forest has been created
        if not self.is_trained():
            self._initialize_trees()

        # we fit all the trees using the new sample
        for tree in self:
            tree.learn_one(x, y)

        self.iteration += 1

        return self

    def predict_proba_one(self, x):
        # Checking that the model has been trained once at least
        # Otherwise return the default empty dict
        if not self.is_trained():
            return {}

        # initialize the scores
        scores = {c: 0 for c in self._classes}

        # Simply computes the prediction for each tree and average it
        for tree in self:
            tree.use_aggregation = self.use_aggregation
            predictions = tree.predict_proba_one(x)
            for c in self._classes:
                scores[c] += predictions[c] / self.n_estimators

        return scores

    @property
    def _multiclass(self):
        return True
