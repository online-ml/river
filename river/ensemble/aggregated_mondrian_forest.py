from river.tree.mondrian.mondrian_tree_classifier import MondrianTreeClassifier

from abc import ABC

from river.base.classifier import Classifier


class AMFLearner(ABC):
    """
    Base class for Aggregated Mondrian Forest classifier and regressors for online learning.

    Note
    ----
    This class is not intended for end users but for development only.

    """

    def __init__(
            self,
            n_estimators: int,
            step: float,
            loss: str,
            use_aggregation: bool,
            split_pure: bool,
            random_state: int,
    ):
        """Instantiates a `AMFLearner` instance.

        Parameters
        ----------
        n_estimators : int
            The number of trees in the forest.

        step : float
            Step-size for the aggregation weights.

        loss : str
            The loss used for the computation of the aggregation weights.

        use_aggregation : bool
            Controls if aggregation is used in the trees. It is highly recommended to
            leave it as `True`.

        split_pure : bool
            Controls if nodes that contains only sample of the same class should be
            split ("pure" nodes). Default is `False`, namely pure nodes are not split,
            but `True` can be sometimes better.

        random_state : int or None
            Controls the randomness involved in the trees.

        """

        # This is yet to be defined by the dataset since we need to know about the amount of features namely
        self._forest = None
        self._n_features = None

        self.n_estimators = n_estimators
        self.step = step
        self.loss = loss
        self.use_aggregation = use_aggregation
        self.split_pure = split_pure
        self.random_state = random_state

    def is_trained(self) -> bool:
        """
        Says whether the model has been trained at least once before.
        Returns
        -------
        trained: bool
        """
        return self._forest is not None

    def check_features_consistency(self, x: dict):
        """
        Makes sure that the features are consistent and set it to the first encountered value is no standard is set.
        Parameters
        ----------
        x: dict
            Feature vector

        Returns
        -------

        """
        n_features = len(list(x.keys()))
        if self._n_features is None:
            self._n_features = n_features
        elif self._n_features != n_features:
            raise Exception("Number of features must be consistent during learning")


class AMFClassifier(AMFLearner, Classifier):
    """
    Aggregated Mondrian Forest classifier for online learning. This algorithm
    is truly online, in the sense that a single pass is performed, and that predictions
    can be produced anytime.

    Each node in a tree predicts according to the distribution of the labels
    it contains. This distribution is regularized using a "Jeffreys" prior
    with parameter ``dirichlet``. For each class with `count` labels in the
    node and `n_samples` samples in it, the prediction of a node is given by

        (count + dirichlet) / (n_samples + dirichlet * n_classes)

    The prediction for a sample is computed as the aggregated predictions of all the
    subtrees along the path leading to the leaf node containing the sample. The
    aggregation weights are exponential weights with learning rate ``step`` and log-loss
    when ``use_aggregation`` is ``True``.

    This computation is performed exactly thanks to a context tree weighting algorithm.
    More details can be found in the paper cited in references below.

    The final predictions are the average class probabilities predicted by each of the
    ``n_estimators`` trees in the forest.

    Note
    ----
    All the parameters of ``AMFClassifier`` become **read-only** after the first call
    to ``learn_one``

    References
    ----------
    J. Mourtada, S. Gaiffas and E. Scornet, *AMF: Aggregated Mondrian Forests for Online Learning*, arXiv:1906.10529, 2019

    """

    def __init__(
            self,
            n_classes: int,
            n_estimators: int = 10,
            step: float = 1.0,
            use_aggregation: bool = True,
            dirichlet: float = None,
            split_pure: bool = False,
            random_state: int = None,
    ):
        """Instantiates a `AMFClassifier` instance.

        Parameters
        ----------
        n_classes : :obj:`int`
            Number of expected classes in the labels. This is required since we
            don't know the number of classes in advance in a online setting.

        n_estimators : :obj:`int`, default = 10
            The number of trees in the forest.

        step : :obj:`float`, default = 1
            Step-size for the aggregation weights. Default is 1 for classification with
            the log-loss, which is usually the best choice.

        loss : {"log"}, default = "log"
            The loss used for the computation of the aggregation weights. Only "log"
            is supported for now, namely the log-loss for multi-class
            classification.

        use_aggregation : :obj:`bool`, default = `True`
            Controls if aggregation is used in the trees. It is highly recommended to
            leave it as `True`.

        dirichlet : :obj:`float` or :obj:`None`, default = `None`
            Regularization level of the class frequencies used for predictions in each
            node. Default is dirichlet=0.5 for n_classes=2 and dirichlet=0.01 otherwise.

        split_pure : :obj:`bool`, default = `False`
            Controls if nodes that contains only sample of the same class should be
            split ("pure" nodes). Default is `False`, namely pure nodes are not split,
            but `True` can be sometimes better.

        random_state : :obj:`int` or :obj:`None`, default = `None`
            Controls the randomness involved in the trees.
        """
        super().__init__(
            n_estimators=n_estimators,
            step=step,
            loss="log",
            use_aggregation=use_aggregation,
            split_pure=split_pure,
            random_state=random_state
        )

        self.n_classes = n_classes
        if dirichlet is None:
            if self.n_classes == 2:
                self.dirichlet = 0.5
            else:
                self.dirichlet = 0.01
        else:
            self.dirichlet = dirichlet

        self._classes = set(range(n_classes))

    def _initialize_trees(self):
        """
        Initialize the forest
        Returns
        -------

        """

        if self._n_features is None:
            raise Exception("You can't initialize the forest without knowning the number of features of the problem. "
                            "Please learn a data point first.")

        self.iteration = 0
        self._forest = [
            MondrianTreeClassifier(
                self.n_classes,
                self._n_features,
                self.step,
                self.use_aggregation,
                self.dirichlet,
                self.split_pure,
                self.iteration,
            )
            for _ in range(self.n_estimators)
        ]

    def learn_one(self, x: dict, y: int):
        """
        Learns the sample (x, y)

        Parameters
        ----------
        x: dict
            Feature vector
        y: int
            Label

        Returns
        -------
        AMFClassifier
        """

        # Checking that y is defined nicely for the trees computation
        if not isinstance(y, int) or y < 0:
            raise Exception("Labels must be positive integers.")

        # Checking the features consistency
        self.check_features_consistency(x)

        # Checking if the forest has been created
        if self._forest is None:
            self._initialize_trees()
        # we fit all the trees using the new sample
        for tree in self._forest:
            tree.learn_one(x, y)
        self.iteration += 1
        return self

    def predict_proba_one(self, x: dict) -> dict[int, float]:
        """
        Predicts the probability of each class for the sample x

        Parameters
        ----------
        x: dict
            Feature vector

        Returns
        -------
        scores: dict
        """

        # Checking that the model has been trained once at least
        if not self.is_trained():
            raise Exception("No sample has been learnt yet. You need to train your model before making predictions.")

        # initialize the scores
        scores = {}
        for j in range(self.n_classes):
            scores[j] = 0

        # Simply computes the prediction for each tree and average it
        for tree in self._forest:
            tree.use_aggregation = self.use_aggregation
            predictions = tree.predict_proba_one(x)
            for j in range(self.n_classes):
                scores[j] += predictions[j] / self.n_estimators

        return scores
