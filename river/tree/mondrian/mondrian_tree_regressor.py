from __future__ import annotations

from river import base
from river._river_rust.tree import NativeMondrianTreeRegressor
from river.tree._numeric_adapter import NumericAdapter
from river.tree.mondrian.mondrian_tree import MondrianTree


class MondrianTreeRegressor(MondrianTree, base.Regressor):
    """Mondrian Tree Regressor.

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
    iteration
        Number iterations to do during training.
    max_nodes
        Maximum number of nodes allowed in the tree. No new splits will occur once this
        limit is reached. If `None`, the tree grows without bound.
    seed
        Random seed for reproducibility.

    Notes
    -----
    The Mondrian Tree Regressor is a type of decision tree that bases splitting decisions over a
    Mondrian process.

    References
    ----------
    [^1]: [Lakshminarayanan, B., Roy, D.M. and Teh, Y.W., 2014. Mondrian forests: Efficient online random forests. Advances in neural information processing systems, 27. (pages 2-4)](https://proceedings.neurips.cc/paper_files/paper/2014/hash/195c9c0797f42473f2c2f922c4cf52cf-Abstract.html)

    """

    def __init__(
        self,
        step: float = 1.0,
        use_aggregation: bool = True,
        iteration: int = 0,
        max_nodes: int | None = None,
        seed: int | None = None,
    ):
        super().__init__(
            step=step,
            loss="least-squares",
            use_aggregation=use_aggregation,
            iteration=iteration,
            max_nodes=max_nodes,
            seed=seed,
        )
        self._native_model: NativeMondrianTreeRegressor | None = None
        self._native_adapter: NumericAdapter | None = None

    @property
    def _is_initialized(self):
        """Check if the tree has learnt at least one sample"""
        return self.iteration != 0

    def _init_native(self, x, y):
        self._native_adapter = NumericAdapter(x, y, sort_features=True)
        self._native_model = NativeMondrianTreeRegressor(len(x), self._rng.getrandbits(64))

    def learn_one(self, x, y):
        if self._native_model is None:
            self._init_native(x, y)
        adapter = self._native_adapter
        model = self._native_model
        assert adapter is not None and model is not None
        self._n_nodes += model.learn_one_mapping(
            x,
            adapter.features,
            adapter.feature_names,
            y,
            self.iteration,
            self.step,
            self.use_aggregation,
            self.max_nodes,
        )
        self.iteration += 1

    def predict_one(self, x):
        """Predict the label of the samples.

        Parameters
        ----------
        x
            Feature vector.

        """

        if self._native_model is None:
            return 0.0

        adapter = self._native_adapter
        assert adapter is not None and self._native_model is not None
        return self._native_model.predict_mapping(x, adapter.features, self.use_aggregation)
