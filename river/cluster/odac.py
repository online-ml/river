from __future__ import annotations

import collections
import itertools
import math
import typing

from river import base, stats


class ODAC(base.Clusterer):
    """The Online Divisive-Agglomerative Clustering (ODAC) aims at continuously
    maintaining a hierarchical cluster structure from evolving time series data
    streams.

    ODAC continuosly monitors the evolution of clusters' diameters and split or merge
    them by gathering more data or reacting to concept drift. Such changes are supported
    by a confidence level that comes from the Hoeffding bound. ODAC relies on keeping
    the linear correlation between series to evaluate whether or not the time series
    hierarchy has changed.

    The distance between time-series a and b is given by `rnomc(a, b) = sqrt((1 - corr(a, b)) / 2)`,
    where `corr(a, b)` is the Pearson Correlation coefficient.

    In the next topics, ε stands for the Hoeffding bound.

    **The Merge Operator**

    The Splitting Criteria guarantees that cluster's diameters monotonically decrease.

    - Assume Clusters: cj with descendants ck and cs.

    - If diameter (ck) - diameter (cj) > ε OR diameter (cs) - diameter (cj ) > ε:

        * There is a change in the correlation structure: Merge clusters ck and cs into cj.

    **Splitting Criteria**

    Consider:

    - d0: the minimum distance;

    - d1: the farthest distance;

    - d2: the second farthest distance.

    Then:

    - if d1 - d2 > εk or t > εk then

        - if (d1 - d0)|(d1 - d_avg) - (d_avg - d0) > εk then

            * Split the cluster


    Parameters
    ----------
    confidence_level
        The confidence level that user wants to work.
    n_min
        Number of minimum observations to gather before checking whether or not
        clusters must be split or merged.
    tau
        The value of tau to use in the calculation of the Hoeffding bound.

    Attributes
    ----------
    structure_changed : bool
        This variable is true when the structure changed, produced by splitting or aggregation.

    Examples
    --------

    >>> from river.cluster import ODAC
    >>> from river.datasets import synth

    >>> model = ODAC(confidence_level=0.9)

    >>> dataset = synth.FriedmanDrift(drift_type='gra', position=(150, 200), seed=42)

    >>> for i, (x, _) in enumerate(dataset.take(500)):
    ...     model.learn_one(x)
    ...     if model.structure_changed:
    ...         print(f"Structure changed at observation {i + 1}")
    Structure changed at observation 1
    Structure changed at observation 100
    Structure changed at observation 200
    Structure changed at observation 300

    >>> print(model.draw(n_decimal_places = 2))
    ROOT d1=0.79 d2=0.76 [NOT ACTIVE]
    ├── CH1_LVL_1 d1=0.74 d2=0.72 [NOT ACTIVE]
    │   ├── CH1_LVL_2 d1=<Not calculated> [3]
    │   └── CH2_LVL_2 d1=0.73 [2, 4]
    └── CH2_LVL_1 d1=0.81 d2=0.78 [NOT ACTIVE]
        ├── CH1_LVL_2 d1=0.73 d2=0.67 [NOT ACTIVE]
        │   ├── CH1_LVL_3 d1=0.72 [0, 9]
        │   └── CH2_LVL_3 d1=<Not calculated> [1]
        └── CH2_LVL_2 d1=0.74 d2=0.73 [NOT ACTIVE]
            ├── CH1_LVL_3 d1=0.71 [5, 6]
            └── CH2_LVL_3 d1=0.71 [7, 8]
    <BLANKLINE>


    You can acess some properties of the clustering model directly:

    >>> model.n_clusters
    11

    >>> model.n_active_clusters
    6

    >>> model.height
    3

    These properties are also available in a summarized form:

    >>> model.summary
    {'n_clusters': 11, 'n_active_clusters': 6, 'height': 3}

    References
    ----------
    [^1]: [Hierarchical clustering of time-series data streams.](http://doi.org/10.1109/TKDE.2007.190727)

    """

    def __init__(self, confidence_level=0.9, n_min=100, tau=0.1):
        if not (confidence_level > 0.0 and confidence_level < 1.0):
            raise ValueError("confidence_level must be between 0 and 1.")
        if not n_min > 0:
            raise ValueError("n_min must be greater than 1.")
        if not tau > 0.0:
            raise ValueError("tau must be greater than 0.")

        self._root_node = ODACCluster("ROOT")
        self.confidence_level = confidence_level
        self.n_min = n_min
        self.tau = tau

        self._update_timer: int = n_min
        self._n_observations: int = 0
        self._is_init = False

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

    def _calculate_height(self, node: ODACCluster) -> int:
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
        if node.active:
            return 1
        elif node.children is not None:
            return sum(
                self._count_active_clusters(child)
                for child in [node.children.first, node.children.second]
            )
        else:
            return 0

    def _find_all_active_clusters(self, node: ODACCluster):
        if node.active:
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

        if not self._is_init:
            # Initialize the first cluster which is the ROOT cluster
            self._root_node(list(x.keys()))
            self._structure_changed = True
            self._is_init = True

        # Update the total observations received
        self._n_observations += 1

        # Split control
        self._update_timer -= 1

        # For each active cluster update the statistics and time to time verify if
        # the cluster needs to merge or expand
        for leaf in self._find_all_active_clusters(self._root_node):
            # For safety
            if not leaf.active:
                continue

            # Update statistics
            leaf.update_statistics(x)

            # Time to time approach
            if self._update_timer == 0:
                # Calculate all the crucial variables to the next procedure
                leaf.calculate_coefficients(self.confidence_level)

                if leaf.test_aggregate() or leaf.test_split(tau=self.tau):
                    # Put the flag change_detected to true to indicate to the user that the structure changed
                    self._structure_changed = True

        # Reset the timer
        if self._update_timer == 0:
            self._update_timer = self.n_min

    # This algorithm does not predict anything. It builds a hierarchical cluster's structure
    def predict_one(self, x: dict):
        raise NotImplementedError

    def draw(self, n_decimal_places: int = 2) -> str:
        """Method to draw the the hierarchical cluster's structure.

        Parameters
        ----------
        n_decimal_places
            The number of decimal places that user wants to view in distances of each active cluster
            in the hierarchical cluster's structure.

        """
        if not (n_decimal_places > 0 and n_decimal_places < 10):
            raise ValueError("n_decimal_places must be between 1 and 9")

        return self._root_node.design_structure(n_decimal_places)

    @property
    def structure_changed(self) -> bool:
        return self._structure_changed


class ODACCluster(base.Base):
    """Cluster class for representing individual clusters."""

    # Constructor method for Cluster class
    def __init__(self, name: str, parent: ODACCluster | None = None):
        self.name = name
        self.parent: ODACCluster | None = parent
        self.active = True
        self.children: ODACChildren | None = None

        self.timeseries_names: list[typing.Hashable]
        self._statistics: dict[tuple[typing.Hashable, typing.Hashable], stats.PearsonCorr] | None

        self.d1: float | None = None
        self.d2: float | None = None
        self.e: float = 0
        self.d0: float | None = None
        self.avg: float | None = None

        self.pivot_0: tuple[typing.Hashable, typing.Hashable]
        self.pivot_1: tuple[typing.Hashable, typing.Hashable]
        self.pivot_2: tuple[typing.Hashable, typing.Hashable]

        self.n = 0

    # Method to design the structure of the cluster tree
    def design_structure(self, decimal_places=2) -> str:
        pre_0 = "    "
        pre_1 = "│   "
        pre_2 = "├── "
        pre_3 = "└── "
        node = self
        prefix = (
            pre_2
            if node.parent is not None and id(node) != id(node.parent.children.second)  # type: ignore
            else pre_3
        )
        while node.parent is not None and node.parent.parent is not None:
            if id(node.parent) != id(node.parent.parent.children.second):  # type: ignore
                prefix = pre_1 + prefix
            else:
                prefix = pre_0 + prefix
            node = node.parent

        if self.d1 is not None:
            r_d1 = f"{self.d1:.{decimal_places}f}"
        else:
            r_d1 = "<Not calculated>"

        if self.d2 is not None:
            r_d2 = f" d2={self.d2:.{decimal_places}f}"
        else:
            r_d2 = ""

        if self.parent is None:
            representation = f"{self.name} d1={r_d1}{r_d2}"
        else:
            representation = f"{prefix}{self.name} d1={r_d1}{r_d2}"

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

    def _init_stats(self) -> dict[tuple[typing.Hashable, typing.Hashable], stats.PearsonCorr]:
        return collections.defaultdict(
            stats.PearsonCorr,
            {
                (k1, k2): stats.PearsonCorr()
                for k1, k2 in itertools.combinations(self.timeseries_names, 2)
            },
        )

    # TODO: not sure if this is the best design
    def __call__(self, ts_names: list[typing.Hashable]):
        """Method that associates the time-series into the cluster and initiates the statistics."""
        self.timeseries_names = sorted(ts_names)  # type: ignore
        self._statistics = self._init_stats()

    def update_statistics(self, x: dict) -> None:
        # For each pair of time-series in the cluster update the correlation
        # values with the data received
        for (k1, k2), item in self._statistics.items():  # type: ignore
            if x.get(k1, None) is None or x.get(k2, None) is None:
                continue
            item.update(float(x[k1]), float(x[k2]))

        # Increment the number of observation in the cluster
        self.n += 1

    # Method to calculate the rnomc values of the cluster
    def _calculate_rnomc_dict(self)-> dict[tuple[typing.Hashable, typing.Hashable], float]:
        # Get the correlation values between time-series in the cluster
        rnomc_dict = {}

        for k1, k2 in itertools.combinations(self.timeseries_names, 2):
            rnomc_dict[(k1, k2)] = math.sqrt((1 - self._statistics[(k1, k2)].get()) / 2.0)  # type: ignore

        return rnomc_dict

    # Method to calculate coefficients for splitting or aggregation
    def calculate_coefficients(self, confidence_level: float) -> None:
        # Get the rnomc values
        rnomc_dict = self._calculate_rnomc_dict()

        if len(rnomc_dict) > 0:
            # Get the average distance in the cluster
            self.avg = sum(rnomc_dict.values()) / self.n

            # Get the minimum distance and the pivot associated in the cluster
            self.pivot_0, self.d0 = min(rnomc_dict.items(), key=lambda x: x[1])
            # Get the maximum distance and the pivot associated in the cluster
            self.pivot_1, self.d1 = max(rnomc_dict.items(), key=lambda x: x[1])

            # Get the second maximum distance and the pivot associated in the cluster
            remaining = {k: v for k, v in rnomc_dict.items() if k != self.pivot_1}

            if len(remaining) > 0:
                self.pivot_2, self.d2 = max(remaining.items(), key=lambda x: x[1])
            else:
                self.pivot_2 = self.d2 = None  # type: ignore

            # Calculate the Hoeffding bound in the cluster
            self.e = math.sqrt(math.log(1 / confidence_level) / (2 * self.n))

    def _get_closest_cluster(self, pivot_1, pivot_2, current, rnormc_dict) -> int:
        """Method that gives the closest cluster where the current time series is located."""

        dist_1 = rnormc_dict.get((min(pivot_1, current), max(pivot_1, current)), 0)
        dist_2 = rnormc_dict.get((min(pivot_2, current), max(pivot_2, current)), 0)
        return 2 if dist_1 >= dist_2 else 1

    def _split_this_cluster(self, pivot_1: typing.Hashable, pivot_2: typing.Hashable, rnormc_dict: dict):
        """Expand into two clusters."""
        pivot_set = {pivot_1, pivot_2}
        pivot_1_list = [pivot_1]
        pivot_2_list = [pivot_2]

        # For each time-series in the cluster we need to find the closest pivot, to then associate with it
        for ts_name in self.timeseries_names:
            if ts_name not in pivot_set:
                cluster = self._get_closest_cluster(pivot_1, pivot_2, ts_name, rnormc_dict)
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

        # Set the active flag to false. Since this cluster is not an active cluster anymore.
        self.active = False
        self.avg = self.d0 = self.pivot_0 = self.pivot_1 = self.pivot_2 = None  # type: ignore
        self._statistics = None

    # Method that proceeds to merge on this cluster
    def _aggregate_this_cluster(self):
        # Reset statistics
        self._statistics = self._init_stats()

        # Put the active flag to true. Since this cluster is an active cluster once again.
        self.active = True
        # Delete and disassociate the children.
        if self.children is not None:
            self.children.reset_parent()
            self.children = None
        # Reset the number of observations in this cluster
        self.n = 0

    # Method to test if the cluster should be split
    def test_split(self, tau: float):
        # Test if the cluster should be split based on specified tau
        if self.d2 is not None:
            if ((self.d1 - self.d2) > self.e) or (tau > self.e):  # type: ignore
                if ((self.d1 - self.d0) * abs(self.d1 + self.d0 - 2 * self.avg)) > self.e:  # type: ignore
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
    """Children class representing child clusters."""

    def __init__(self, first: ODACCluster, second: ODACCluster):
        self.first = first
        self.second = second

    def reset_parent(self):
        self.first.parent = None
        self.second.parent = None
