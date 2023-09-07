from __future__ import annotations

import functools

from river import anomaly, utils
from river.neighbors.base import DistanceFunc
from river.utils import VectorDict


class LocalOutlierFactor(anomaly.base.AnomalyDetector):
    """Incremental Local Outlier Factor (Incremental LOF).

    Incremental LOF Algorithm as described in the reference paper

    The Incremental Local Outlier Factor (ILOF) is an online version of the Local Outlier Factor (LOF) used to identify outliers based on density of local neighbors.

    We consider:
        - NewPoints: new points;
        - kNN(p): the neighboors of p (the k closest points to p)
        - RkNN(p): the rev-neighboors of p (points that have p as one of their neighboors)
        - Set_upd_lrd: Set of points that need to update the local reachability distance
        - Set_upd_lof: Set of points that need to update the local outlier factor

    The algorithm here implemented based on the original one in the paper is:
        1) Insert NewPoints and calculate its distance to existing points
        2) Update the neighboors and reverse-neighboors of all the points
        3) Define sets of affected points that required update
        4) Calculate the reachability-distance from new point to neighboors (NewPoints -> kNN(NewPoints)) and from rev-neighboors to new point (RkNN(NewPoints) -> NewPoints)
        5) Update the reachability-distance for affected points: RkNN(RkNN(NewPoints)) -> RkNN(NewPoints)
        6) Update local reachability distance of affected points: lrd(Set_upd_lrd)
        7) Update local outlier factor: lof(Set_upd_lof)

    Parameters
    ----------
    n_neighbors
        The number of nearest neighbors to use for density estimation.
    window_size
        The size of the batch of data to be taken in at once for the model to learn
    distance_func
        Distance function to be used. By default, the Euclidean distance is used.
    verbose
        Whether or not to print messages

    Attributes
    ----------
    X
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
    skip_first
        A boolean value indicating whether to skip the first window of data.

    Example
    ----------

    >>> from river import anomaly
    >>> from river import datasets
    >>> import pandas as pd

    >>> cc_df = pd.DataFrame(datasets.CreditCard())

    >>> k = 20 # Define number of nearest neighbors
    >>> incremental_lof = anomaly.LocalOutlierFactor(k, verbose=False)

    >>> for x, _ in datasets.CreditCard().take(200):
    ...    incremental_lof.learn_one(x)

    >>> ilof_scores = []
    >>> for x in cc_df[0][201:206]:
    ...    ilof_scores.append(incremental_lof.score_one(x))

    >>> [round(ilof_score, 3) for ilof_score in ilof_scores]
    [1.149, 1.098, 1.158, 1.101, 1.092]

    References
    ----------
    Pokrajac, David & Lazarevic, Aleksandar & Latecki, Longin Jan. (2007). Incremental Local Outlier Detection for Data Streams. Proceedings of the 2007 IEEE Symposium on Computational Intelligence and Data Mining, CIDM 2007. 504-515. 10.1109/CIDM.2007.368917.
    """

    def __init__(
        self,
        n_neighbors: int = 10,
        verbose=True,
        distance_func: DistanceFunc = None,
    ):
        self.n_neighbors = n_neighbors
        self.X: list = []
        self.x_batch: list = []
        self.x_scores: list = []
        self.dist_dict: dict = {}
        self.neighborhoods: dict = {}
        self.rev_neighborhoods: dict = {}
        self.k_dist: dict = {}
        self.reach_dist: dict = {}
        self.lof: dict = {}
        self.local_reach: dict = {}
        self.verbose = verbose
        self.distance = (
            distance_func
            if distance_func is not None
            else functools.partial(utils.math.minkowski_distance, p=2)
        )

    def learn_one(self, x: dict):
        """
        Update the model with one incoming observation

        Parameters
        ----------
        x
            A dictionary of feature values.
        """
        self.x_batch.append(x)
        if len(self.X) or len(self.x_batch) > 1:
            self.learn(self.x_batch)
            self.x_batch = []

    def learn(self, x_batch: list):
        x_batch, equal = self.check_equal(x_batch, self.X)
        if equal != 0 and self.verbose:
            print("At least one sample is equal to previously observed instances.")

        if len(x_batch) == 0:
            if self.verbose:
                print("No new data was added.")
        else:
            # Increase size of objects to acomodate new data
            (
                nm,
                self.X,
                self.neighborhoods,
                self.rev_neighborhoods,
                self.k_dist,
                self.reach_dist,
                self.dist_dict,
                self.local_reach,
                self.lof,
            ) = self.expand_objects(
                x_batch,
                self.X,
                self.neighborhoods,
                self.rev_neighborhoods,
                self.k_dist,
                self.reach_dist,
                self.dist_dict,
                self.local_reach,
                self.lof,
            )

            # Calculate neighborhoods, reverse neighborhoods, k-distances and distances between neighboors
            (
                self.neighborhoods,
                self.rev_neighborhoods,
                self.k_dist,
                self.dist_dict,
            ) = self.initial_calculations(
                self.X, nm, self.neighborhoods, self.rev_neighborhoods, self.k_dist, self.dist_dict
            )

            # Define sets of particles
            (
                Set_new_points,
                Set_neighbors,
                Set_rev_neighbors,
                Set_upd_lrd,
                Set_upd_lof,
            ) = self.define_sets(nm, self.neighborhoods, self.rev_neighborhoods)

            # Calculate new reachability distance of all affected points
            self.reach_dist = self.calc_reach_dist_newpoints(
                Set_new_points,
                self.neighborhoods,
                self.rev_neighborhoods,
                self.reach_dist,
                self.dist_dict,
                self.k_dist,
            )
            self.reach_dist = self.calc_reach_dist_otherpoints(
                Set_rev_neighbors,
                self.neighborhoods,
                self.rev_neighborhoods,
                self.reach_dist,
                self.dist_dict,
                self.k_dist,
            )

            # Calculate new local reachability distance of all affected points
            self.local_reach = self.calc_local_reach_dist(
                Set_upd_lrd, self.neighborhoods, self.reach_dist, self.local_reach
            )

            # Calculate new Local Outlier Factor of all affected points
            self.lof = self.calc_lof(Set_upd_lof, self.neighborhoods, self.local_reach, self.lof)

    def score_one(self, x: dict):
        """
        Score a new incoming observation based on model constructed previously.
        Perform same calculations as 'learn_one' function but doesn't add the new calculations to the atributes
        Data samples that are equal to samples stored by the model are not considered.

        Parameters
        ----------
        x
            A dictionary of feature values.

        Returns
        -------
        lof : list
            List of LOF calculated for incoming data
        """

        self.x_scores.append(x)

        self.x_scores, equal = self.check_equal(self.x_scores, self.X)
        if equal != 0 and self.verbose:
            print("The new observation is the same to one of the previously observed instances.")

        if len(self.x_scores) == 0:
            if self.verbose:
                print("No new data was added.")
        else:
            Xs = self.X.copy()
            (
                nm,
                Xs,
                neighborhoods,
                rev_neighborhoods,
                k_dist,
                reach_dist,
                dist_dict,
                local_reach,
                lof,
            ) = self.expand_objects(
                self.x_scores,
                Xs,
                self.neighborhoods,
                self.rev_neighborhoods,
                self.k_dist,
                self.reach_dist,
                self.dist_dict,
                self.local_reach,
                self.lof,
            )

            neighborhoods, rev_neighborhoods, k_dist, dist_dict = self.initial_calculations(
                Xs, nm, neighborhoods, rev_neighborhoods, k_dist, dist_dict
            )
            (
                Set_new_points,
                Set_neighbors,
                Set_rev_neighbors,
                Set_upd_lrd,
                Set_upd_lof,
            ) = self.define_sets(nm, neighborhoods, rev_neighborhoods)
            reach_dist = self.calc_reach_dist_newpoints(
                Set_new_points, neighborhoods, rev_neighborhoods, reach_dist, dist_dict, k_dist
            )
            reach_dist = self.calc_reach_dist_otherpoints(
                Set_rev_neighbors,
                neighborhoods,
                rev_neighborhoods,
                reach_dist,
                dist_dict,
                k_dist,
            )
            local_reach = self.calc_local_reach_dist(
                Set_upd_lrd, neighborhoods, reach_dist, local_reach
            )
            lof = self.calc_lof(Set_upd_lof, neighborhoods, local_reach, lof)
            self.x_scores = []

            return lof[nm[0]]

    def initial_calculations(
        self,
        X: list,
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
        X
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

        # Calculate distances all particles consdering new and old ones
        new_distances = [
            [i, j, self.distance(X[i], X[j])] for i in range(n + m) for j in range(i) if i >= n
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

    @staticmethod
    def check_equal(x_list: list, y_list: list):
        """Check if new list of observations (x_list) has any data sample that is equal to
           any previous data recorded (y_list)."""
        result = [x for x in x_list if not any(x == y for y in y_list)]
        return result, len(x_list) - len(result)

    def expand_objects(
        self,
        new_particles: list,
        X: list,
        neighborhoods: dict,
        rev_neighborhoods: dict,
        k_dist: dict,
        reach_dist: dict,
        dist_dict: dict,
        local_reach: dict,
        lof: dict,
    ):
        """Expand size of dictionaries and lists to fit new data"""
        n = len(X)
        m = len(new_particles)
        X.extend(new_particles)
        neighborhoods.update({i: [] for i in range(n + m)})
        rev_neighborhoods.update({i: [] for i in range(n + m)})
        k_dist.update({i: float("inf") for i in range(n + m)})
        reach_dist.update({i + n: {} for i in range(m)})
        dist_dict.update({i + n: {} for i in range(m)})
        local_reach.update({i + n: [] for i in range(m)})
        lof.update({i + n: [] for i in range(m)})
        return (
            (n, m),
            X,
            neighborhoods,
            rev_neighborhoods,
            k_dist,
            reach_dist,
            dist_dict,
            local_reach,
            lof,
        )

    def define_sets(self, nm, neighborhoods: dict, rev_neighborhoods: dict):
        """Define sets of points for the incremental LOF algorithm"""
        # Define set of new points from batch
        Set_new_points = set(range(nm[0], nm[0] + nm[1]))
        Set_neighbors: set = set()
        Set_rev_neighbors: set = set()

        # Define neighbors and reverse neighbors of new data points
        for i in Set_new_points:
            Set_neighbors = set(Set_neighbors) | set(neighborhoods[i])
            Set_rev_neighbors = set(Set_rev_neighbors) | set(rev_neighborhoods[i])

        # Define points that need to update their local reachability distance because of new data points
        Set_upd_lrd = Set_rev_neighbors
        for j in Set_rev_neighbors:
            Set_upd_lrd = Set_upd_lrd | set(rev_neighborhoods[j])
        Set_upd_lrd = Set_upd_lrd | Set_new_points

        # Define points that need to update their lof because of new data points
        Set_upd_lof = Set_upd_lrd
        for m in Set_upd_lrd:
            Set_upd_lof = Set_upd_lof | set(rev_neighborhoods[m])
        Set_upd_lof = Set_upd_lof

        return Set_new_points, Set_neighbors, Set_rev_neighbors, Set_upd_lrd, Set_upd_lof

    def calc_reach_dist_newpoints(
        self,
        Set: set,
        neighborhoods: dict,
        rev_neighborhoods: dict,
        reach_dist: dict,
        dist_dict: dict,
        k_dist: dict,
    ):
        """Calculate reachability distance from new points to neighbors and from neighbors to new points"""
        for c in Set:
            for j in set(neighborhoods[c]):
                reach_dist[c][j] = max(dist_dict[c][j], k_dist[j])
            for j in set(rev_neighborhoods[c]):
                reach_dist[j][c] = max(dist_dict[j][c], k_dist[c])
        return reach_dist

    def calc_reach_dist_otherpoints(
        self,
        Set: set,
        neighborhoods: dict,
        rev_neighborhoods: dict,
        reach_dist: dict,
        dist_dict: dict,
        k_dist: dict,
    ):
        """Calculate reachability distance from reverse neighbors of reverse neighbors ( RkNN(RkNN(NewPoints)) ) to reverse neighbors ( RkNN(NewPoints) )
        These values change because of the insertion of new points"""
        for j in Set:
            for i in set(rev_neighborhoods[j]):
                reach_dist[i][j] = max(dist_dict[i][j], k_dist[j])
        return reach_dist

    def calc_local_reach_dist(
        self, Set: set, neighborhoods: dict, reach_dist: dict, local_reach_dist: dict
    ):
        """Calculate local reachability distance of affected points"""
        for i in Set:
            local_reach_dist[i] = len(neighborhoods[i]) / sum(
                [reach_dist[i][j] for j in neighborhoods[i]]
            )
        return local_reach_dist

    def calc_lof(self, Set: set, neighborhoods: dict, local_reach: dict, lof: dict):
        """Calculate local outlier factor of affected points"""
        for i in Set:
            lof[i] = sum([local_reach[j] for j in neighborhoods[i]]) / (
                len(neighborhoods[i]) * local_reach[i]
            )
        return lof
