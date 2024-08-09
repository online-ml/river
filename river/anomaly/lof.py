from __future__ import annotations

import copy
import functools

import pandas as pd

from river import anomaly, utils
from river.neighbors.base import DistanceFunc


def check_equal(x_list: list, y_list: list):
    """
    Check if new list of observations (x_list) has any data sample that is equal to any previous data recorded (y_list).
    """
    result = [x for x in x_list if not any(x == y for y in y_list)]
    return result, len(x_list) - len(result)


def expand_objects(
    new_particles: list,
    x_list: list,
    neighborhoods: dict,
    rev_neighborhoods: dict,
    k_dist: dict,
    reach_dist: dict,
    dist_dict: dict,
    local_reach: dict,
    lof: dict,
):
    """
    Expand size of dictionaries and lists to take into account new data points.
    """
    n = len(x_list)
    m = len(new_particles)
    x_list.extend(new_particles)
    neighborhoods.update({i: [] for i in range(n + m)})
    rev_neighborhoods.update({i: [] for i in range(n + m)})
    k_dist.update({i: float("inf") for i in range(n + m)})
    reach_dist.update({i + n: {} for i in range(m)})
    dist_dict.update({i + n: {} for i in range(m)})
    local_reach.update({i + n: [] for i in range(m)})
    lof.update({i + n: [] for i in range(m)})
    return (
        (n, m),
        x_list,
        neighborhoods,
        rev_neighborhoods,
        k_dist,
        reach_dist,
        dist_dict,
        local_reach,
        lof,
    )


def define_sets(nm, neighborhoods: dict, rev_neighborhoods: dict):
    """
    Define sets of points for the incremental LOF algorithm.
    """
    # Define set of new points from batch
    set_new_points = set(range(nm[0], nm[0] + nm[1]))
    set_neighbors: set = set()
    set_rev_neighbors: set = set()

    # Define neighbors and reverse neighbors of new data points
    for i in set_new_points:
        set_neighbors = set(set_neighbors) | set(neighborhoods[i])
        set_rev_neighbors = set(set_rev_neighbors) | set(rev_neighborhoods[i])

    # Define points that need to update their local reachability distance because of new data points
    set_upd_lrd = set_rev_neighbors
    for j in set_rev_neighbors:
        set_upd_lrd = set_upd_lrd | set(rev_neighborhoods[j])
    set_upd_lrd = set_upd_lrd | set_new_points

    # Define points that need to update their lof because of new data points
    set_upd_lof = set_upd_lrd
    for m in set_upd_lrd:
        set_upd_lof = set_upd_lof | set(rev_neighborhoods[m])
    set_upd_lof = set_upd_lof

    return set_new_points, set_neighbors, set_rev_neighbors, set_upd_lrd, set_upd_lof


def calc_reach_dist_new_points(
    set_index: set,
    neighborhoods: dict,
    rev_neighborhoods: dict,
    reach_dist: dict,
    dist_dict: dict,
    k_dist: dict,
):
    """
    Calculate reachability distance from new points to neighbors and from neighbors to new points.
    """
    for c in set_index:
        for j in set(neighborhoods[c]):
            reach_dist[c][j] = max(dist_dict[c][j], k_dist[j])
        for j in set(rev_neighborhoods[c]):
            reach_dist[j][c] = max(dist_dict[j][c], k_dist[c])
    return reach_dist


def calc_reach_dist_other_points(
    set_index: set,
    rev_neighborhoods: dict,
    reach_dist: dict,
    dist_dict: dict,
    k_dist: dict,
):
    """
    Calculate reachability distance from reverse neighbors of reverse neighbors ( RkNN(RkNN(NewPoints)) )
    to reverse neighbors ( RkNN(NewPoints) ). These values change due to the insertion of new points.
    """
    for j in set_index:
        for i in set(rev_neighborhoods[j]):
            reach_dist[i][j] = max(dist_dict[i][j], k_dist[j])
    return reach_dist


def calc_local_reach_dist(
    set_index: set, neighborhoods: dict, reach_dist: dict, local_reach_dist: dict
):
    """
    Calculate local reachability distance of affected points.
    """
    for i in set_index:
        denominator = sum(reach_dist[i][j] for j in neighborhoods[i])
        local_reach_dist[i] = len(neighborhoods[i]) / denominator if denominator else 0
    return local_reach_dist


def calc_lof(set_index: set, neighborhoods: dict, local_reach: dict, lof: dict):
    """
    Calculate local outlier factor (LOF) of affected points.
    """
    for i in set_index:
        denominator = len(neighborhoods[i]) * local_reach[i]
        lof[i] = sum(local_reach[j] for j in neighborhoods[i]) / denominator if denominator else 0
    return lof


class LocalOutlierFactor(anomaly.base.AnomalyDetector):
    """Incremental Local Outlier Factor (Incremental LOF).

    The Incremental Local Outlier Factor (ILOF) is an online version of the Local Outlier Factor (LOF), proposed by
    Pokrajac et al. (2017),  and is used to identify outliers based on density of local neighbors.

    The algorithm take into account the following elements:
        - `NewPoints`: new points;
        - `kNN(p)`: the k-nearest neighboors of `p` (the k-closest points to `p`);
        - `RkNN(p)`: the reverse-k-nearest neighboors of `p` (points that have `p` as one of their neighboors);
        - `set_upd_lrd`: Set of points that need to have the local reachability distance updated;
        - `set_upd_lof`: Set of points that need to have the local outlier factor updated.

    This current implementation within `River`, based on the original one in the paper, follows the following steps:
        1) Insert new data points (`NewPoints`) and calculate its distance to existing points;
        2) Update the nreaest neighboors and reverse nearest neighboors of all the points;
        3) Define sets of affected points that required updates;
        4) Calculate the reachability-distance from new point to neighboors (`NewPoints` -> `kNN(NewPoints)`)
           and from rev-neighboors to new point (`RkNN(NewPoints)` -> `NewPoints`);
        5) Update the reachability-distance for affected points: `RkNN(RkNN(NewPoints))` -> `RkNN(NewPoints)`
        6) Update local reachability distance of affected points: `lrd(set_upd_lrd)`;
        7) Update local outlier factor: `lof(set_upd_lof)`.

    The incremental LOF algorithm is expected to provide equivalent detection performance as the iterated static
    LOF algroithm (applied after insertion of each data record), while requiring significantly less computational time.
    Moreover, the insertion of a new data point as well as deletion of an old data point influence only a limited number
    of their closest neighbors, which means that the number of updates per such insertion/deletion does not depend
    on the total number of instances learned/in the data set.

    Parameters
    ----------
    n_neighbors
        The number of nearest neighbors to use for density estimation.
    distance_func
        Distance function to be used. By default, the Euclidean distance is used.

    Attributes
    ----------
    x_list
        A list of stored observations.
    x_batch
        A buffer to hold incoming observations until it's time to update the model.
    x_scores
        A buffer to hold incoming observations until it's time to score them.
    dist_dict
        A dictionary to hold distances between observations.
    neighborhoods
        A dictionary to hold neighborhoods for each observation.
    rev_neighborhoods
        A dictionary to hold reverse neighborhoods for each observation.
    k_dist
        A dictionary to hold k-distances for each observation.
    reach_dist
        A dictionary to hold reachability distances for each observation.
    lof
        A dictionary to hold Local Outlier Factors for each observation.
    local_reach
        A dictionary to hold local reachability distances for each observation.

    Examples
    --------

    >>> import pandas as pd
    >>> from river import anomaly
    >>> from river import datasets

    >>> cc_df = pd.DataFrame(datasets.CreditCard())

    >>> lof = anomaly.LocalOutlierFactor(n_neighbors=20)

    >>> for x, _ in datasets.CreditCard().take(200):
    ...     lof.learn_one(x)

    >>> lof.learn_many(cc_df[201:401])

    >>> scores = []
    >>> for x in cc_df[0][401:406]:
    ...     scores.append(lof.score_one(x))

    >>> [round(score, 3) for score in scores]
    [1.802, 1.936, 1.566, 1.181, 1.272]

    >>> X = [0.5, 0.45, 0.43, 0.44, 0.445, 0.45, 0.0]
    >>> lof = anomaly.LocalOutlierFactor()

    >>> for x in X[:3]:
    ...     lof.learn_one({'x': x})  # Warming up

    >>> for x in X:
    ...     features = {'x': x}
    ...     print(
    ...         f'Anomaly score for x={x:.3f}: {lof.score_one(features):.3f}')
    ...     lof.learn_one(features)
    Anomaly score for x=0.500: 0.000
    Anomaly score for x=0.450: 0.000
    Anomaly score for x=0.430: 0.000
    Anomaly score for x=0.440: 1.020
    Anomaly score for x=0.445: 1.032
    Anomaly score for x=0.450: 0.000
    Anomaly score for x=0.000: 0.980

    References
    ----------
    David Pokrajac, Aleksandar Lazarevic, and Longin Jan Latecki (2007). Incremental Local Outlier Detection for Data
    Streams. In: Proceedings of the 2007 IEEE Symposium on Computational Intelligence and Data Mining (CIDM 2007). 504-515.
    DOI: 10.1109/CIDM.2007.368917.

    """

    def __init__(
        self,
        n_neighbors: int = 10,
        distance_func: DistanceFunc | None = None,
    ):
        self.n_neighbors = n_neighbors
        self.x_list: list = []
        self.x_batch: list = []
        self.x_scores: list = []
        self.dist_dict: dict = {}
        self.neighborhoods: dict = {}
        self.rev_neighborhoods: dict = {}
        self.k_dist: dict = {}
        self.reach_dist: dict = {}
        self.lof: dict = {}
        self.local_reach: dict = {}
        self.distance_func = distance_func
        self.distance = (
            distance_func
            if distance_func is not None
            else functools.partial(utils.math.minkowski_distance, p=2)
        )

    def learn_many(self, x: pd.DataFrame):
        x = x[0].tolist()
        self.learn(x)

    def learn_one(self, x: dict):
        self.x_batch.append(x)
        if len(self.x_list) or len(self.x_batch) > 1:
            self.learn(self.x_batch)
            self.x_batch = []

    def learn(self, x_batch: list):
        x_batch, equal = check_equal(x_batch, self.x_list)

        # Increase size of objects to accommodate new data
        (
            nm,
            self.x_list,
            self.neighborhoods,
            self.rev_neighborhoods,
            self.k_dist,
            self.reach_dist,
            self.dist_dict,
            self.local_reach,
            self.lof,
        ) = expand_objects(
            x_batch,
            self.x_list,
            self.neighborhoods,
            self.rev_neighborhoods,
            self.k_dist,
            self.reach_dist,
            self.dist_dict,
            self.local_reach,
            self.lof,
        )

        # Calculate neighborhoods, reverse neighborhoods, k-distances and distances between neighbors
        (
            self.neighborhoods,
            self.rev_neighborhoods,
            self.k_dist,
            self.dist_dict,
        ) = self._initial_calculations(
            self.x_list,
            nm,
            self.neighborhoods,
            self.rev_neighborhoods,
            self.k_dist,
            self.dist_dict,
        )

        # Define sets of particles
        (
            set_new_points,
            set_neighbors,
            set_rev_neighbors,
            set_upd_lrd,
            set_upd_lof,
        ) = define_sets(nm, self.neighborhoods, self.rev_neighborhoods)

        # Calculate new reachability distance of all affected points
        self.reach_dist = calc_reach_dist_new_points(
            set_new_points,
            self.neighborhoods,
            self.rev_neighborhoods,
            self.reach_dist,
            self.dist_dict,
            self.k_dist,
        )
        self.reach_dist = calc_reach_dist_other_points(
            set_rev_neighbors,
            self.rev_neighborhoods,
            self.reach_dist,
            self.dist_dict,
            self.k_dist,
        )

        # Calculate new local reachability distance of all affected points
        self.local_reach = calc_local_reach_dist(
            set_upd_lrd, self.neighborhoods, self.reach_dist, self.local_reach
        )

        # Calculate new Local Outlier Factor of all affected points
        self.lof = calc_lof(set_upd_lof, self.neighborhoods, self.local_reach, self.lof)

    def score_one(self, x: dict):
        self.x_scores.append(x)
        self.x_scores, equal = check_equal(self.x_scores, self.x_list)

        if len(self.x_scores) == 0 or len(self.x_list) == 0:
            return 0.0

        x_list_copy = self.x_list.copy()

        (
            nm,
            x_list_copy,
            neighborhoods,
            rev_neighborhoods,
            k_dist,
            reach_dist,
            dist_dict,
            local_reach,
            lof,
        ) = expand_objects(
            self.x_scores,
            x_list_copy,
            self.neighborhoods.copy(),
            self.rev_neighborhoods.copy(),
            self.k_dist.copy(),
            copy.deepcopy(self.reach_dist),
            copy.deepcopy(self.dist_dict),
            self.local_reach.copy(),
            self.lof.copy(),
        )

        neighborhoods, rev_neighborhoods, k_dist, dist_dict = self._initial_calculations(
            x_list_copy, nm, neighborhoods, rev_neighborhoods, k_dist, dist_dict
        )
        (
            set_new_points,
            set_neighbors,
            set_rev_neighbors,
            set_upd_lrd,
            set_upd_lof,
        ) = define_sets(nm, neighborhoods, rev_neighborhoods)
        reach_dist = calc_reach_dist_new_points(
            set_new_points, neighborhoods, rev_neighborhoods, reach_dist, dist_dict, k_dist
        )
        reach_dist = calc_reach_dist_other_points(
            set_rev_neighbors,
            rev_neighborhoods,
            reach_dist,
            dist_dict,
            k_dist,
        )
        local_reach = calc_local_reach_dist(set_upd_lrd, neighborhoods, reach_dist, local_reach)
        lof = calc_lof(set_upd_lof, neighborhoods, local_reach, lof)
        self.x_scores = []

        # Use nm[0] as index since upon this configuration nm[1] is expected to be 1.
        return lof[nm[0]]

    def _initial_calculations(
        self,
        x_list: list,
        nm: tuple,
        neighborhoods: dict,
        rev_neighborhoods: dict,
        k_distances: dict,
        dist_dict: dict,
    ):
        """
        Perform initial calculations on the incoming data before applying the Incremental LOF algorithm.
        Taking the new data, it updates the neighborhoods, reverse neighborhoods, k-distances and distances between particles.

        Parameters
        ----------
        x_list
            A list of stored observations.
        nm
            A tuple representing the current size of the dataset.
        neighborhoods
            A dictionary of particle neighborhoods.
        rev_neighborhoods
            A dictionary of reverse particle neighborhoods.
        k_distances
            A dictionary to hold k-distances for each observation.
        dist_dict
            A dictionary of dictionaries storing distances between particles

        Returns
        -------
        neighborhoods
            Updated dictionary of particle neighborhoods
        rev_neighborhoods
            Updated dictionary of reverse particle neighborhoods
        k_distances
            Updated dictionary to hold k-distances for each observation
        dist_dict
            Updated dictionary of dictionaries storing distances between particles

        """

        n = nm[0]
        m = nm[1]
        k = self.n_neighbors

        # Calculate distances all particles considering new and old ones
        new_distances = [
            [i, j, self.distance(x_list[i], x_list[j])]
            for i in range(n + m)
            for j in range(i)
            if i >= n
        ]
        # Add new distances to distance dictionary
        for i in range(len(new_distances)):
            dist_dict[new_distances[i][0]][new_distances[i][1]] = new_distances[i][2]
            dist_dict[new_distances[i][1]][new_distances[i][0]] = new_distances[i][2]

        # Calculate new k-dist for each particle
        for i, inner_dict in enumerate(dist_dict.values()):
            k_distances[i] = sorted(inner_dict.values())[min(k, len(inner_dict.values())) - 1]

        # Only keep particles that are neighbors in distance dictionary
        dist_dict = {
            k: {k2: v2 for k2, v2 in v.items() if v2 <= k_distances[k]}
            for k, v in dist_dict.items()
        }

        # Define new neighborhoods for particles
        for key, value in dist_dict.items():
            neighborhoods[key] = [index for index in value]

        # Define new reverse neighborhoods for particles
        for particle_id, neighbor_ids in neighborhoods.items():
            for neighbor_id in neighbor_ids:
                rev_neighborhoods[neighbor_id].append(particle_id)

        return neighborhoods, rev_neighborhoods, k_distances, dist_dict
