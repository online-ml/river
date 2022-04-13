from .neighbors import DistanceFunc, MinkowskiNeighbors


class BaseKNN:
    """Base neighbors class.

    Base neighbors class to make shared functionality for instantiating the
    nearest neighbors. We also provide this so that the model classes are
    able to be pickled.

    Parameters
    ----------

    n_neighbors
        Number of neighbors to use.

    window_size
        Size of the sliding window use to search neighbors with.

    min_distance_keep
        The minimum distance (similarity) to consider adding a point to the window.
        E.g., a value of 0.0 will add even exact duplicates. Default is 0.05 to add
        similar but not exactly the same points.

    distance_func
        An optional distance function that should be a callable with two inputs
        of Any type (e.g., a comparison A against B) that returns a distance.
        If not defined, the default Minkowski distance is used.

    """

    def __init__(
        self,
        n_neighbors: int = 5,
        window_size: int = 1000,
        min_distance_keep: float = 0.0,
        distance_func: DistanceFunc = None,
    ):
        # These are not needed here, but the CI will fail without them
        self.n_neighbors = n_neighbors
        self.window_size = window_size
        self.min_distance_keep = min_distance_keep
        self.distance_func = distance_func

        self.nn = MinkowskiNeighbors(
            window_size=window_size,
            distance_func=distance_func,
            min_distance_keep=min_distance_keep,
            n_neighbors=n_neighbors,
        )
