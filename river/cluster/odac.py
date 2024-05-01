from __future__ import annotations

import collections
import itertools
import math

from river import base, stats


class ODAC(base.Clusterer):
    """ODAC: The Online Divisive-Agglomerative Clustering (ODAC) is an
    algorithm whose the goal is to continuously maintain a hierarchical cluster's
    structure from evolving time series data streams.

    * Performs hierarchical clustering
    * Continuously monitor the evolution of clusters' diameters
    * Two Operators:
        * Splitting: expand the structure more data, more detailed clusters
        * Merge: contract the structure reacting to changes.
    * Splitting and agglomerative criteria are supported by a confidence
        level given by the Hoeffding bounds.

    Main Algorithm

    * ForEver
        * Read Next Example
        * For all the active clusters
            * Update the sufficient statistics
        * Time to Time
            * Verify Merge Clusters
            * Verify Expand Cluster

    Feeding ODAC

    * Each example is processed once.
    * Only sufficient statistics at leaves are updated.
    * Sufficient Statistics: a triangular matrix of the correlations between
        variables in a leaf.

    Similarity Distance

    * Distance between time-series (a & b): rnomc(a,b) = sqrt((1-corr(a,b))/2) where
        corr(a,b) is the Pearson Correlation coefficient.

    The Merge Operator

    The Splitting Criteria guarantees that cluster's diameters monotonically decrease.
        * Assume Clusters: cj with descendants ck and cs.
        * If diameter (ck) - diameter (cj) > ε OR diameter (cs) - diameter (cj ) > ε:
            * Change in the correlation structure!
            * Merge clusters ck and cs into cj.

    Splitting Criteria

    d1 = d(a,b) the farthest distance
    d2 the second farthest distance

    if d1 - d2 > εk or t > εk then
        if (d1 - d0)|(d1 - d_avg) - (d_avg - d0) > εk then
            * Split the cluster

    Parameters
    ----------
    confidence_level
        The confidence level that user wants to work.
    n_min
        Number of minimum observations to time to time verify Merge or Expand Clusters.
    tau
        The value of tau to use in the calculation of the Hoeffding bound.

    Attributes
    ----------
    structure_changed : bool
        This variable is true when the structure changed, produced by splitting or aggregation.

    Examples
    --------

    In the following example we utilize one dataset described in paper [^1].

    >>> from river.cluster import ODAC
    >>> from river.datasets import synth

    >>> model = ODAC(confidence_level=0.9, n_min=50)

    >>> dataset = synth.FriedmanDrift(drift_type='lea',position=(1, 2, 3),seed=42)

    >>> for i, (X, _) in enumerate(dataset.take(300)):
    ...     model.learn_one(X)
    ...     if model.structure_changed:
    ...         print("#################")
    ...         print(f"Change detected at observation {i}")
    ...         model.draw(n_decimal_places = 2)
    #################
    Change detected at observation 0
    ROOT d1=<Not calculated> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    #################
    Change detected at observation 49
    ROOT d1=0.83 d2=0.82 [NOT ACTIVE]
    ├── CH1_LVL_1 d1=<Not calculated> [0, 1, 4, 5, 6, 7, 8]
    └── CH2_LVL_1 d1=<Not calculated> [2, 3, 9]
    #################
    Change detected at observation 99
    ROOT d1=0.83 d2=0.82 [NOT ACTIVE]
    ├── CH1_LVL_1 d1=0.79 d2=0.78 [NOT ACTIVE]
    │   ├── CH1_LVL_2 d1=<Not calculated> [0, 4, 5]
    │   └── CH2_LVL_2 d1=<Not calculated> [1, 6, 7, 8]
    └── CH2_LVL_1 d1=0.79 d2=0.74 [NOT ACTIVE]
        ├── CH1_LVL_2 d1=<Not calculated> [2, 9]
        └── CH2_LVL_2 d1=<Not calculated> [3]
    #################
    Change detected at observation 149
    ROOT d1=0.83 d2=0.82 [NOT ACTIVE]
    ├── CH1_LVL_1 d1=0.79 d2=0.78 [NOT ACTIVE]
    │   ├── CH1_LVL_2 d1=0.74 d2=0.71 [NOT ACTIVE]
    │   │   ├── CH1_LVL_3 d1=<Not calculated> [0]
    │   │   └── CH2_LVL_3 d1=<Not calculated> [4, 5]
    │   └── CH2_LVL_2 d1=0.80 d2=0.77 [NOT ACTIVE]
    │       ├── CH1_LVL_3 d1=<Not calculated> [1, 6]
    │       └── CH2_LVL_3 d1=<Not calculated> [7, 8]
    └── CH2_LVL_1 d1=0.79 d2=0.74 [NOT ACTIVE]
        ├── CH1_LVL_2 d1=0.74 [2, 9]
        └── CH2_LVL_2 d1=<Not calculated> [3]
    #################
    Change detected at observation 199
    ROOT d1=0.83 d2=0.82 [NOT ACTIVE]
    ├── CH1_LVL_1 d1=0.79 d2=0.78 [NOT ACTIVE]
    │   ├── CH1_LVL_2 d1=0.74 d2=0.71 [0, 4, 5]
    │   └── CH2_LVL_2 d1=0.80 d2=0.77 [NOT ACTIVE]
    │       ├── CH1_LVL_3 d1=0.76 [1, 6]
    │       └── CH2_LVL_3 d1=0.67 [7, 8]
    └── CH2_LVL_1 d1=0.79 d2=0.74 [NOT ACTIVE]
        ├── CH1_LVL_2 d1=0.70 [2, 9]
        └── CH2_LVL_2 d1=<Not calculated> [3]
    #################
    Change detected at observation 249
    ROOT d1=0.83 d2=0.82 [NOT ACTIVE]
    ├── CH1_LVL_1 d1=0.79 d2=0.78 [NOT ACTIVE]
    │   ├── CH1_LVL_2 d1=0.79 d2=0.66 [NOT ACTIVE]
    │   │   ├── CH1_LVL_3 d1=<Not calculated> [0, 4]
    │   │   └── CH2_LVL_3 d1=<Not calculated> [5]
    │   └── CH2_LVL_2 d1=0.80 d2=0.77 [NOT ACTIVE]
    │       ├── CH1_LVL_3 d1=0.75 [1, 6]
    │       └── CH2_LVL_3 d1=0.66 [7, 8]
    └── CH2_LVL_1 d1=0.79 d2=0.74 [NOT ACTIVE]
        ├── CH1_LVL_2 d1=0.69 [2, 9]
        └── CH2_LVL_2 d1=<Not calculated> [3]

    >>> model.draw()
    ROOT d1=0.8336 d2=0.8173 [NOT ACTIVE]
    ├── CH1_LVL_1 d1=0.7927 d2=0.7762 [NOT ACTIVE]
    │   ├── CH1_LVL_2 d1=0.7871 d2=0.6562 [NOT ACTIVE]
    │   │   ├── CH1_LVL_3 d1=0.7558 [0, 4]
    │   │   └── CH2_LVL_3 d1=<Not calculated> [5]
    │   └── CH2_LVL_2 d1=0.8014 d2=0.7718 [NOT ACTIVE]
    │       ├── CH1_LVL_3 d1=0.7459 [1, 6]
    │       └── CH2_LVL_3 d1=0.6663 [7, 8]
    └── CH2_LVL_1 d1=0.7853 d2=0.7428 [NOT ACTIVE]
        ├── CH1_LVL_2 d1=0.6908 [2, 9]
        └── CH2_LVL_2 d1=<Not calculated> [3]

    >>> print("n_clusters = {}".format(model.n_clusters))
    n_clusters = 11

    >>> print("n_active_clusters = {}".format(model.n_active_clusters))
    n_active_clusters = 6

    >>> print("height = {}".format(model.height))
    height = 3

    >>> print("summary = {}".format(model.summary))
    summary = {'n_clusters': 11, 'n_active_clusters': 6, 'height': 3}

    References
    ----------
    [^1]: [Hierarchical clustering of time-series data streams.](http://doi.org/10.1109/TKDE.2007.190727)

    """

    def __init__(self, confidence_level=0.9, n_min=100, tau=0.1):
        if not (confidence_level > 0.0 and confidence_level < 1.0):
            raise ValueError("confidence_level must be between 0 and 1")
        if not n_min > 0:
            raise ValueError("n_min must be greater than 1")
        if not tau > 0.0:
            raise ValueError("tau must be greater than 0")

        self._root_node = ODACCluster("ROOT")
        self.confidence_level = confidence_level
        self.n_min = n_min
        self.tau = tau

        self._observations_threshold = n_min
        self._n_observations = 0

        self._structure_changed = False

    @property
    def n_active_clusters(self):
        return self._count_active_clusters(self._root_node)

    @property
    def n_clusters(self):
        return self._count_clusters(self._root_node)

    @property
    def height(self) -> int:
        return self._calculate_height(self._root_node)

    @property
    def summary(self):
        summary = {
            "n_clusters": self.n_clusters,
            "n_active_clusters": self.n_active_clusters,
            "height": self.height,
        }
        return summary

    def _calculate_height(self, node: ODACCluster):
        if node.children is not None:
            child_heights = [
                self._calculate_height(child)
                for child in [node.children.first, node.children.second]
            ]
            return max(child_heights) + 1
        else:
            return 0

    def _count_clusters(self, node: ODACCluster) -> int:
        count = 1
        if node.children is not None:
            for child in [node.children.first, node.children.second]:
                count += self._count_clusters(child)
        return count

    def _count_active_clusters(self, node: ODACCluster) -> int:
        if node.active is True:
            return 1
        elif node.children is not None:
            return sum(
                self._count_active_clusters(child)
                for child in [node.children.first, node.children.second]
            )
        else:
            return 0

    def _find_all_active_clusters(self, node: ODACCluster):
        if node.active is True:
            yield node
        elif node.children is not None:
            for child in [node.children.first, node.children.second]:
                yield from self._find_all_active_clusters(child)

    def learn_one(self, x: dict):
        # If x is empty, do nothing
        if not x:
            return

        if self._structure_changed:
            self._structure_changed = False

        # Auxiliar variable
        pass_threshold = False

        if self._n_observations == 0:
            # Initialize the first cluster wich is the ROOT cluster
            self._root_node(list(x.keys()))
            self._structure_changed = True

        # Update the total observations received
        self._n_observations += 1

        # For each active cluster update the statistics and time to time verify if the cluster needs to merge or expand
        for leaf in self._find_all_active_clusters(self._root_node):
            # For safety
            if not leaf.active:
                continue

            # Update statistics
            leaf.update_statistics(x)

            # Time to time approach
            if self._n_observations >= self._observations_threshold:
                # Calculate all the crucial variables to the next procedure
                leaf.calculate_coefficients(self.confidence_level)

                if leaf.test_aggregate() or leaf.test_split(tau=self.tau):
                    # Put the flag change_detected to true to indicate to the user that the structure changed
                    self._structure_changed = True

                if pass_threshold is False:
                    pass_threshold = True
        if pass_threshold is True:
            self._observations_threshold += self.n_min

    # This algorithm does not predict anything. It builds a hierarchical cluster's structure
    def predict_one(self, x: dict):
        raise NotImplementedError

    def draw(self, n_decimal_places=4) -> None:
        """Method to draw the the hierarchical cluster's structure.

        Parameters
        ----------
        n_decimal_places
            The number of decimal places that user wants to view in distances of each active cluster
            in the hierarchical cluster's structure.

        """
        if not (n_decimal_places > 0 and n_decimal_places < 10):
            raise ValueError("n_decimal_places must be between 1 and 9")
        print(self._root_node.design_structure(n_decimal_places), end="")

    @property
    def structure_changed(self) -> bool:
        return self._structure_changed


class ODACCluster(base.Base):
    """Cluster class for representing individual clusters"""

    # Constructor method for Cluster class
    def __init__(self, name: str, parent: ODACCluster | None = None):
        self.active = True
        self.name = name
        self.parent: ODACCluster = parent
        self.children: ODACChildren = None

        self.d1: float = None
        self.d2: float = None

        self.e: float = 0.0

        self.d0: float = None
        self.avg: float = None

        self.pivot_0: tuple = None
        self.pivot_1: tuple = None
        self.pivot_2: tuple = None

        self.n = 0

    # Method to design the structure of the cluster tree
    def design_structure(self, decimal_places=4) -> str:
        pre_0 = "    "
        pre_1 = "│   "
        pre_2 = "├── "
        pre_3 = "└── "
        node = self
        prefix = (
            pre_2
            if node.parent is not None and id(node) != id(node.parent.children.second)
            else pre_3
        )
        while node.parent is not None and node.parent.parent is not None:
            if id(node.parent) != id(node.parent.parent.children.second):
                prefix = pre_1 + prefix
            else:
                prefix = pre_0 + prefix
            node = node.parent

        # TODO create auxiliary variables in order to shorten the lines with "representation = " and make them more readable
        if self.parent is None:
            representation = f"{self.name} d1={'{:.{dp}f}'.format(self.d1, dp=decimal_places) if self.d1 is not None else '<Not calculated>'}{' d2=' + '{:.{dp}f}'.format(self.d2, dp=decimal_places) if self.d2 is not None else ''}"
        else:
            representation = f"{prefix}{self.name} d1={'{:.{dp}f}'.format(self.d1, dp=decimal_places) if self.d1 is not None else '<Not calculated>'}{' d2=' + '{:.{dp}f}'.format(self.d2, dp=decimal_places) if self.d2 is not None else ''}"
        if self.active is True:
            return representation + f" {self.timeseries_names}\n"
        else:
            representation += " [NOT ACTIVE]\n"
        # Do the structure recursively
        if self.children is not None:
            for child in [self.children.first, self.children.second]:
                representation += child.design_structure(decimal_places)
        return representation

    def __str__(self) -> str:
        return self.design_structure()

    def __repr__(self) -> str:
        return self.design_structure()

    # Method that associates the time-series into the cluster and initiates the statistics using PearsonCorr
    def __call__(self, ts_names: list[str]):
        self.timeseries_names: list[str] = sorted(ts_names)
        self._statistics = collections.defaultdict(
            stats.PearsonCorr,
            {
                (k1, k2): stats.PearsonCorr()
                for k1, k2 in itertools.combinations(self.timeseries_names, 2)
            },
        )

    # Method to update the statistics of the cluster
    def update_statistics(self, x: dict) -> None:
        # If x is empty, do nothing
        if not x:
            return

        # For each pair of time-series in the cluster update the correlation values with the data received
        for (k1, k2), item in self._statistics.items():
            if x.get(k1, None) is None or x.get(k2, None) is None:
                continue
            item.update(float(x[k1]), float(x[k2]))

        # Increment the number of observation in the cluster
        self.n += 1

    # Method to get the correlation values of the cluster
    def _get_correlation_dict(self) -> dict:
        corr_dict = {}

        for (k1, k2), item in self._statistics.items():
            corr_dict[(k1, k2)] = item.get()

        return corr_dict

    # Method to calculate the rnomc values of the cluster
    def _calculate_rnomc_dict(self) -> dict:
        rnomc_dict = {}
        # Get the correlation values between time-series in the cluster
        corr_dict = self._get_correlation_dict()

        for i, k1 in enumerate(self.timeseries_names):
            for k2 in self.timeseries_names[i + 1 :]:
                rnomc_dict[(k1, k2)] = math.sqrt((1 - corr_dict[(k1, k2)]) / 2)

        return rnomc_dict

    # Method to calculate coefficients for splitting or aggregation
    def calculate_coefficients(self, confidence_level: float) -> None:
        # Get the rnomc values
        rnomc_dict = self._calculate_rnomc_dict()

        if bool(rnomc_dict):
            # Get the average distance in the cluster
            self.avg = sum(rnomc_dict.values()) / self.n

            # Get the minimum distance and the pivot associated in the cluster
            self.pivot_0, self.d0 = min(rnomc_dict.items(), key=lambda x: x[1])
            # Get the maximum distance and the pivot associated in the cluster
            self.pivot_1, self.d1 = max(rnomc_dict.items(), key=lambda x: x[1])

            # Get the second maximum distance and the pivot associated in the cluster
            remaining = {k: v for k, v in rnomc_dict.items() if k != self.pivot_1}
            if bool(remaining):
                self.pivot_2, self.d2 = max(remaining.items(), key=lambda x: x[1])
            else:
                self.pivot_2 = self.d2 = None

            r_sqrd = 1  # Let's consider this value
            # Calculate the Hoeffding bound in the cluster
            self.e = math.sqrt(r_sqrd * math.log(1 / confidence_level) / (2 * self.n))

        elif None not in [
            self.avg,
            self.d0,
            self.pivot_0,
            self.d1,
            self.pivot_1,
            self.d2,
            self.pivot_2,
            self.e,
        ]:
            self.avg = (
                self.d0
            ) = self.pivot_0 = self.d1 = self.pivot_1 = self.d2 = self.pivot_2 = self.e = None

    # Method that gives the closer cluster that the current time series is
    def _get_closer_cluster(
        self, pivot_1: str, pivot_2: str, current: str, rnormc_dict: dict
    ) -> int:
        dist_1 = rnormc_dict.get((min(pivot_1, current), max(pivot_1, current)), 0)
        dist_2 = rnormc_dict.get((min(pivot_2, current), max(pivot_2, current)), 0)
        return 2 if dist_1 >= dist_2 else 1

    # Method that procedes to expanding this cluster into two clusters
    def _split_this_cluster(self, pivot_1: str, pivot_2: str, rnormc_dict: dict) -> None:
        pivot_set = {pivot_1, pivot_2}
        pivot_1_list = [pivot_1]
        pivot_2_list = [pivot_2]

        # For each time-series in the cluster we need to find the closest pivot, to then associate with it
        for ts_name in self.timeseries_names:
            if ts_name not in pivot_set:
                cluster = self._get_closer_cluster(pivot_1, pivot_2, ts_name, rnormc_dict)
                if cluster == 1:
                    pivot_1_list.append(ts_name)
                else:
                    pivot_2_list.append(ts_name)

        new_name = "1" if self.name == "ROOT" else str(int(self.name.split("_")[-1]) + 1)

        # Create the two new clusters. The children of this cluster
        cluster_child_1 = ODACCluster("CH1_LVL_" + new_name, parent=self)
        cluster_child_1(pivot_1_list)

        cluster_child_2 = ODACCluster("CH2_LVL_" + new_name, parent=self)
        cluster_child_2(pivot_2_list)

        self.children = ODACChildren(cluster_child_1, cluster_child_2)

        # Set the active flag to false. Since this cluster is not an active cluster no more.
        self.active = False
        self.avg = self.d0 = self.pivot_0 = self.pivot_1 = self.pivot_2 = None
        del self._statistics

    # Method that proceeds to merge on this cluster
    def _aggregate_this_cluster(self):
        # Reset statistics
        self._statistics = collections.defaultdict(
            stats.PearsonCorr,
            {
                (k1, k2): stats.PearsonCorr()
                for k1, k2 in itertools.combinations(self.timeseries_names, 2)
            },
        )
        # Put the active flag to true. Since this cluster is an active cluster once again.
        self.active = True
        # Delete and disassociate the children.
        if self.children is not None:
            self.children.reset_parent()
            del self.children
            self.children = None
        # Reset the number of observations in this cluster
        self.n = 0

    # Method to test if the cluster should be split
    def test_split(self, tau: float):
        # Test if the cluster should be split based on specified tau
        if self.d2 is not None:
            if ((self.d1 - self.d2) > self.e) or (tau > self.e):
                if ((self.d1 - self.d0) * abs(self.d1 + self.d0 - 2 * self.avg)) > self.e:
                    # Split this cluster
                    self._split_this_cluster(
                        pivot_1=self.pivot_1[0],
                        pivot_2=self.pivot_1[1],
                        rnormc_dict=self._calculate_rnomc_dict(),
                    )
                    return True
        return False

    # Method to test if the cluster should be aggregated
    def test_aggregate(self):
        # Test if the cluster should be aggregated
        if self.parent is not None and self.d1 is not None:
            if self.d1 - self.parent.d1 > max(self.parent.e, self.e):
                self.parent._aggregate_this_cluster()
                return True
        return False


class ODACChildren(base.Base):
    """Children class representing child clusters"""

    # Constructor method for Children class
    def __init__(self, first: ODACCluster, second: ODACCluster):
        self.first = first
        self.second = second

    # Method to reset the parent of children clusters
    def reset_parent(self) -> None:
        self.first.parent = None
        self.second.parent = None
