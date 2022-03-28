from typing import Callable

from river import base, utils

from .base_neighbors import NearestNeighbors

__all__ = ["KNNClassifier"]


class KNNClassifier(NearestNeighbors, base.Classifier):
    """
    K-Nearest Neighbors (KNN) for classification.

    This works by storing a buffer with the `window_size` most recent observations.
    A brute-force search is used to find the `n_neighbors` nearest observations
    in the buffer to make a prediction. See the NearestNeighbors parent class for model
    details.

    Parameters
    ----------
    n_neighbors
        The number of nearest neighbors to search for.

    window_size
        The maximum size of the window storing the last observed samples.

    min_distance_keep
        The minimum distance (similarity) to consider adding a point to the window.
        E.g., a value of 0.0 will add even exact duplicates. Default is 0.05 to add
        similar but not exactly the same points.

    weighted
        Weight the contribution of each neighbor by it's inverse distance.

    class_cleanup
        Boolean to indicate if you always want to cleanup the list of known
        classes based on the current window. If true, cleanup happens after
        learn. If False, you can call it manually (or not at all). This is ideal
        for models that have a changing and growing number of classes.

    distance_func
        An optional distance function that should accept an a=, b=, and any
        custom set of kwargs (defined in distance_func_kwargs). If not defined,
        the default Minkowski distance is used.

    distance_func_kwargs
        A dictionary to pass as kwargs to your distance function, in addition
        to a= and b=. If distance_func is set to None, these are ignored,
        and are set to including the power parameter for the Minkowski metric.
        For this parameter, when `p=1`, this corresponds to the Manhattan
        distance, while `p=2` corresponds to the Euclidean distance.

    Notes
    -----
    See the NearestNeighbors documentation for details about the base model. Note that
    since the window is moving and we keep track of all classes that are added
    at some point, a class might be returned in a result (with a value of 0)
    if it is no longer in the window. You can call model.class_cleanup() if
    you want to iterate through points to ensure no extra classes are present.

    Examples
    --------
    >>> from river import datasets, neighbors, preprocessing
    >>> from river import evaluate, metrics
    >>> dataset = datasets.Phishing()

    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     neighbors.KNNClassifier()
    ... )

    >>> for x, y in dataset.take(100):
    ...     model = model.learn_one(x, y)

    >>> for x, y in dataset.take(1):
    ...     model.predict_one(x)
    {False: 0.0, True: 1.0}
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        window_size: int = 1000,
        min_distance_keep: float = 0.0,
        weighted: bool = True,
        class_cleanup: bool = False,
        distance_func: Callable = None,
        distance_func_kwargs: dict = None,
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            window_size=window_size,
            min_distance_keep=min_distance_keep,
            distance_func=distance_func,
            distance_func_kwargs=distance_func_kwargs,
        )
        self.weighted = weighted
        self.class_cleanup = class_cleanup
        self.reset()

    def reset(self) -> "KNNClassifier":
        """
        Reset estimator
        Returns:
            self
        """
        self._reset()
        self.classes = set()
        return self

    def _class_cleanup(self) -> "KNNClassifier":
        """
        Classes that are added (and removed) from the window may no longer be valid.
        This method iterates through the current window and ensures only known classes
        are added. This comes at a cost of O(N) to loop through entire window.
        Returns:
            self
        """
        self.classes = set()
        [self.classes.add(x) for x in self.window if x[1] is not None]
        return self

    def learn_one(self, x: dict, y=None):
        """Learn a set of features `x` and optional class `y`.
        Parameters:
            x: A dictionary of features.
            y: A class (optional if known).
        Returns:
            self
        """
        # Only add the class y to known classes if we actually add the point!
        if self.update(x, y, n_neighbors=self.n_neighbors):
            self.classes.add(y)

        # Ensure classes known to instance reflect window
        if self.class_cleanup:
            self._class_cleanup()
        return self

    def predict_proba_one(self, x):
        """Predict the class of a set of features `x`.
        Parameters:
            x: A dictionary of features.
        Returns:
            Lookup (dict) of classes and probability predictions (normalized)
        """
        nearest = self.find_nearest(x=x, n_neighbors=self.n_neighbors)

        # Default prediction for every class we know is 0.
        # If class_cleanup is false this can include classes not in window
        y_pred = {c: 0.0 for c in self.classes}

        # No nearest points? Return the default.
        if not nearest:
            return y_pred

        # If the closest is an exact match AND has a class, return it
        if nearest[0][-1] == 0 and nearest[0][1] is not None:

            # Update the class in our prediction from 0 to 1, 100% certain!
            y_pred[nearest[0][1]] = 1.0
            return y_pred

        for neighbor in nearest:
            distance = neighbor[-1]
            y = neighbor[1]

            # Weighted votes by inverse distance
            if self.weighted:
                y_pred[y] += 1.0 / distance

            # Uniform votes
            else:
                y_pred[y] += 1.0

        # Normalize votes into real [0, 1] probabilities
        return utils.math.softmax(y_pred)
