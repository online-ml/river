from __future__ import annotations

import functools

from river import base, utils
from river.neighbors import SWINN

from .base import BaseNN, FunctionWrapper


class KNNClassifier(base.Classifier):
    """K-Nearest Neighbors (KNN) for classification.

    Samples are stored using a first-in, first-out strategy. The strategy to perform search
    queries in the data buffer is defined by the `engine` parameter.

    Parameters
    ----------
    n_neighbors
        The number of nearest neighbors to search for.
    engine
        The search engine used to store the instances and perform search queries. Depending
        on the choose engine, search will be exact or approximate. Please, consult the
        documentation of each available search engine for more details on its usage.
        By default, use the `SWINN` search engine for approximate search queries.
    weighted
        Weight the contribution of each neighbor by it's inverse distance.
    cleanup_every
        This determines at which rate old classes are cleaned up. Classes that
        have been seen in the past but that are not present in the current
        window are dropped. Classes are never dropped when this is set to 0.
    softmax
        Whether or not to use softmax normalization to normalize the neighbors contributions.
        Votes are divided by the total number of votes if this is `False`.

    Notes
    -----
    Note that since the window is moving and we keep track of all classes that
    are added at some point, a class might be returned in a result (with a
    value of 0) if it is no longer in the window. You can call
    model.clean_up_classes(), or set `cleanup_every` to a non-zero value.

    Examples
    --------
    >>> import functools
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import neighbors
    >>> from river import preprocessing
    >>> from river import utils

    >>> dataset = datasets.Phishing()

    To select a custom distance metric which takes one or several parameter, you can wrap your
    chosen distance using `functools.partial`:

    >>> l1_dist = functools.partial(utils.math.minkowski_distance, p=1)

    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     neighbors.KNNClassifier(
    ...         engine=neighbors.SWINN(
    ...             dist_func=l1_dist,
    ...             seed=42
    ...         )
    ...     )
    ... )

    >>> evaluate.progressive_val_score(dataset, model, metrics.Accuracy())
    Accuracy: 89.67%

    """

    def __init__(
        self,
        n_neighbors: int = 5,
        engine: BaseNN | None = None,
        weighted: bool = True,
        cleanup_every: int = 0,
        softmax: bool = False,
    ):
        self.n_neighbors = n_neighbors

        if engine is None:
            engine = SWINN(dist_func=functools.partial(utils.math.minkowski_distance, p=2))

        if not isinstance(engine.dist_func, FunctionWrapper):
            engine.dist_func = FunctionWrapper(engine.dist_func)

        self.engine = engine
        self.weighted = weighted
        self.cleanup_every = cleanup_every
        self.classes: set[base.typing.ClfTarget] = set()
        self.softmax = softmax
        self._cleanup_counter = cleanup_every

        # Create a fresh copy of the supplied search engine
        self._nn: BaseNN = self.engine.clone(include_attributes=True)

    @property
    def _multiclass(self):
        return True

    @classmethod
    def _unit_test_params(cls):
        from river.neighbors import LazySearch

        yield {
            "n_neighbors": 3,
            "engine": LazySearch(
                window_size=30, dist_func=functools.partial(utils.math.minkowski_distance, p=2)
            ),
        }

    def clean_up_classes(self):
        """Clean up classes added to the window.

        Classes that are added (and removed) from the window may no longer be valid.
        This method cleans up the window and and ensures only known classes
        are added, and we do not consider "None" a class. It is called every
        `cleanup_every` step, or can be called manually.

        """
        self.classes = {x for x in self.window if x[0][1] is not None}

    def learn_one(self, x, y):
        # Update the data buffer
        self._nn.append((x, y))

        # Update the set of known classes
        self.classes.add(y)

        # Ensure classes known to instance reflect window
        self._run_class_cleanup()
        return self

    def _run_class_cleanup(self):
        """Helper function to run class cleanup, accounting for _cleanup_counter."""
        # clean up classes every cleanup_every steps
        if self.cleanup_every:
            self._cleanup_counter -= 1
            if self._cleanup_counter == 0:
                self.clean_up_classes()
                self._cleanup_counter = self.cleanup_every

        return self

    def predict_proba_one(self, x, **kwargs):
        nearest = self._nn.search((x, None), n_neighbors=self.n_neighbors, **kwargs)

        # Default prediction for every class we know is 0.
        # If class_cleanup is false this can include classes not in window
        y_pred = {c: 0.0 for c in self.classes}

        # No nearest points? Return the default (normalized)
        # Note that normalization otherwise happens at the end
        if not nearest:
            default_pred = 1 / len(self.classes) if self.classes else 0.0
            return {c: default_pred for c in self.classes}

        neighbors, distances = nearest

        # If the closest is an exact match AND has a class, return it
        if distances[0] == 0 and neighbors[0][1] is not None:
            # Update the class in our prediction from 0 to 1, 100% certain!
            y_pred[neighbors[0][1]] = 1.0
            return y_pred

        for neighbor, distance in zip(neighbors, distances):
            (x, y) = neighbor

            # Weighted votes by inverse distance
            if self.weighted:
                y_pred[y] += 1.0 / distance

            # Uniform votes
            else:
                y_pred[y] += 1.0

        # Normalize votes into real [0, 1] probabilities
        if self.softmax:
            return utils.math.softmax(y_pred)

        # Otherwise normalize by the total sum
        total = sum(y_pred.values())
        for y in y_pred:
            y_pred[y] /= total
        return y_pred
