from __future__ import annotations

from river import base
from river._river_rust.tree import NativeMondrianTreeClassifier
from river.tree._numeric_adapter import NumericAdapter
from river.tree.mondrian.mondrian_tree import MondrianTree


class MondrianTreeClassifier(MondrianTree, base.Classifier):
    """Mondrian Tree classifier.

        By default, this implementation assumes that all feature values are scaled between 0 and 1.
        If you cannot assume the minimum and maximum values for each feature,
        you can use preprocessing.MinMaxScaler as an initial preprocessing step.
        This is important because Mondrian trees are highly sensitive to feature scaling, as
        the distance between a sample and the node's bounding box is calculated as the sum of the distances across all features.

        Parameters
        ----------
        step
            Step of the tree.
        use_aggregation
            Whether to use aggregation weighting techniques or not.
        dirichlet
            Dirichlet parameter of the problem.
        split_pure
            Whether the tree should split pure leafs during training or not.
        iteration
            Number iterations to do during training.
        max_nodes
            Maximum number of nodes allowed in the tree. No new splits will occur once this
            limit is reached. If `None`, the tree grows without bound.
        seed
            Random seed for reproducibility.

        Notes
        -----
        The Mondrian Tree Classifier is a type of decision tree that bases splitting decisions over a
        Mondrian process.

        Examples
        --------
    >>> from river import datasets
        >>> from river import evaluate
        >>> from river import metrics
        >>> from river import preprocessing
        >>> from river import tree

        >>> dataset = datasets.Bananas()

        >>> model = (
        ...     preprocessing.MinMaxScaler() |
        ...     tree.mondrian.MondrianTreeClassifier(
        ...         step=1.0,
        ...         use_aggregation=True,
        ...         dirichlet=0.5,
        ...         seed=1
        ...     )
        ... )

        >>> metric = metrics.Accuracy()

        >>> evaluate.progressive_val_score(dataset, model, metric)
        Accuracy: 72.24%

        References
        ----------
        [^1]: [Lakshminarayanan, B., Roy, D.M. and Teh, Y.W., 2014. Mondrian forests: Efficient online random forests. Advances in neural information processing systems, 27. (pages 2-4)](https://proceedings.neurips.cc/paper_files/paper/2014/hash/195c9c0797f42473f2c2f922c4cf52cf-Abstract.html)

    """

    def __init__(
        self,
        step: float = 1.0,
        use_aggregation: bool = True,
        dirichlet: float = 0.5,
        split_pure: bool = False,
        iteration: int = 0,
        max_nodes: int | None = None,
        seed: int | None = None,
    ):
        super().__init__(
            step=step,
            loss="log",
            use_aggregation=use_aggregation,
            iteration=iteration,
            max_nodes=max_nodes,
            seed=seed,
        )

        self.dirichlet = dirichlet
        self.split_pure = split_pure

        # Training attributes
        # The previously observed classes set
        self._classes: set[base.typing.ClfTarget] = set()

        self._class_to_idx: dict = {}
        self._idx_to_class: list = []
        self._native_class_indices: list[int] = []
        self._native_model: NativeMondrianTreeClassifier | None = None
        self._native_adapter: NumericAdapter | None = None

    @property
    def _is_initialized(self):
        """Check if the tree has learnt at least one sample"""
        return len(self._classes) != 0

    def _init_native(self, x, y):
        adapter = NumericAdapter(x, y, sort_features=True)
        self._native_adapter = adapter
        self._class_to_idx = adapter.labels
        self._idx_to_class = adapter.classes
        self._native_class_indices = [0]
        self._native_model = NativeMondrianTreeClassifier(len(x), 1, self._rng.getrandbits(64))

    @property
    def _multiclass(self):
        return True

    def learn_one(self, x, y):
        if self._native_model is None:
            self._init_native(x, y)
        adapter = self._native_adapter
        model = self._native_model
        assert adapter is not None and model is not None
        y_index, added = adapter.add_label(y)
        if added:
            model.add_class()
            self._native_class_indices.append(y_index)
        self._classes.add(y)
        self._n_nodes += model.learn_one_mapping(
            x,
            adapter.features,
            adapter.feature_names,
            y_index,
            self.iteration,
            self.step,
            self.use_aggregation,
            self.dirichlet,
            self.split_pure,
            self.max_nodes,
        )
        self.iteration += 1

    def predict_proba_one(self, x):
        """Predict the probability of the samples.

        Parameters
        ----------
        x
            Feature vector.

        """

        if not self._is_initialized:
            return {}

        adapter = self._native_adapter
        assert adapter is not None and self._native_model is not None
        return self._native_model.predict_proba_mapping(
            x,
            adapter.features,
            self._idx_to_class,
            self._native_class_indices,
            self.dirichlet,
            self.use_aggregation,
        )
