from __future__ import annotations

import collections.abc

import numpy as np

from river import base
from river.utils.math import mahalanobis_distance

__all__ = ["GNG"]


class _Edge:
    """Undirected topological edge connecting two GNG units."""

    __slots__ = ("p0", "p1", "age")

    def __init__(self, p0: _Unit, p1: _Unit, age: int = 0):
        self.p0 = p0
        self.p1 = p1
        self.age = age


class _Unit:
    """Neuron/prototype unit in a Growing Neural Gas graph."""

    __slots__ = ("id", "w", "error", "neighbors")

    def __init__(self, unit_id: int, w: np.ndarray):
        self.id = unit_id
        self.w = np.array(w, dtype=np.float64)
        self.error = 0.0
        self.neighbors: list[_Edge] = []

    def add_neighbor(self, other: _Unit) -> None:
        edge = _Edge(self, other)
        self.neighbors.append(edge)
        other.neighbors.append(edge)

    def search_edge(self, other: _Unit) -> _Edge | None:
        for edge in self.neighbors:
            if edge.p1 is other or edge.p0 is other:
                return edge
        return None

    def update_ages(self) -> None:
        for edge in self.neighbors:
            edge.age += 1

    def purge_edges(self, max_age: int) -> None:
        i = 0
        while i < len(self.neighbors):
            edge = self.neighbors[i]
            if edge.age >= max_age:
                self.neighbors.pop(i)
                other = edge.p1 if edge.p0 is self else edge.p0
                if edge in other.neighbors:
                    other.neighbors.remove(edge)
            else:
                i += 1

    def remove_neighbor(self, other: _Unit) -> None:
        # Fritzke (1995) Step 8d: undirected graph — remove edge from both endpoints
        self.neighbors = [e for e in self.neighbors if not (e.p0 is other or e.p1 is other)]
        other.neighbors = [e for e in other.neighbors if not (e.p0 is self or e.p1 is self)]

    def get_neighborhood(self) -> list[_Unit]:
        return [e.p0 if e.p0 is not self else e.p1 for e in self.neighbors]


class GNG(base.Clusterer):
    """Growing Neural Gas (GNG) incremental topology learner.

    Growing Neural Gas [^1] learns an undirected topological graph of prototype units
    that adapts incrementally to the data distribution, inserting new units in regions
    of highest accumulated error and pruning aged edges and isolated units.

    Parameters
    ----------
    eps_b
        Learning rate for moving the winning unit `s1` towards the input signal.
    eps_n
        Learning rate for moving the direct topological neighbors of `s1` towards the input.
    max_age
        Maximum age of an edge. Edges reaching `age >= max_age` are removed.
    lambda_
        Number of samples between insertions of a new unit.
    alpha
        Error reduction factor applied to the highest-error unit `q` and its worst neighbor `f`
        upon inserting a new unit.
    d
        Global multiplicative error decay factor applied to all units after each step.
    max_nodes
        Optional maximum number of units in the graph. If `None`, the graph grows unbounded.
    dist_metric
        Distance metric used to compare vectors: `"euclidean"` or `"mahalanobis"`.
    inv_cov
        Precision matrix (inverse covariance) as a 2D `numpy.ndarray` when
        `dist_metric="mahalanobis"`.
    distance_func
        Optional custom callable `(a: np.ndarray, b: np.ndarray) -> float` overriding
        `dist_metric`.

    Attributes
    ----------
    n_samples_seen : int
        Total number of input signals processed.
    n_prototypes_created : int
        Number of interpolated units created via the `lambda_` insertion rule.

    Examples
    --------
    >>> from river import cluster
    >>> gng = cluster.GNG(lambda_=5, max_age=50)
    >>> for i in range(20):
    ...     x = {"x": float(i % 2) * 10.0, "y": float(i % 2) * 10.0}
    ...     gng.learn_one(x)
    >>> gng.n_clusters >= 2
    True
    >>> isinstance(gng.predict_one({"x": 0.0, "y": 0.0}), int)
    True

    References
    ----------
    [^1]: Fritzke, B. (1995). A growing neural gas network learns topologies.
          Advances in Neural Information Processing Systems 7 (NIPS 1994), pp. 625-632.

    """

    def __init__(
        self,
        eps_b: float = 0.2,
        eps_n: float = 0.006,
        max_age: int = 200,
        lambda_: int = 5,
        alpha: float = 0.5,
        d: float = 0.995,
        max_nodes: int | None = None,
        dist_metric: str = "euclidean",
        inv_cov: np.ndarray | None = None,
        distance_func: collections.abc.Callable | None = None,
    ):
        self.eps_b = eps_b
        self.eps_n = eps_n
        self.max_age = max_age
        self.lambda_ = lambda_
        self.alpha = alpha
        self.d = d
        self.max_nodes = max_nodes
        self.dist_metric = dist_metric
        self.inv_cov = inv_cov
        self.distance_func = distance_func

        self._units: list[_Unit] = []
        self._features: list | None = None
        self._feature_set: set = set()
        self._next_id: int = 0
        self.n_samples_seen: int = 0
        self.n_prototypes_created: int = 0

    def _new_unit(self, w: np.ndarray) -> _Unit:
        unit = _Unit(self._next_id, w)
        self._next_id += 1
        return unit

    def _dist(self, w1: np.ndarray, w2: np.ndarray) -> float:
        if self.distance_func is not None:
            return float(self.distance_func(w1, w2))
        if self.dist_metric.lower() == "euclidean" or self.inv_cov is None:
            diff = w1 - w2
            return float(np.sqrt(np.maximum(0.0, np.dot(diff, diff))))
        return mahalanobis_distance(w1, w2, self.inv_cov)

    def _dict_to_array(self, x: dict, allow_new_features: bool = True) -> np.ndarray:
        if self._features is None:
            if not allow_new_features:
                return np.array([float(x[k]) for k in sorted(x)], dtype=np.float64)
            self._features = sorted(x)
            self._feature_set = set(self._features)
        elif allow_new_features:
            new_keys = sorted(k for k in x if k not in self._feature_set)
            if new_keys:
                self._features.extend(new_keys)
                self._feature_set.update(new_keys)
                pad_width = len(new_keys)
                for u in self._units:
                    u.w = np.pad(u.w, (0, pad_width))
        return np.array([float(x.get(k, 0.0)) for k in self._features], dtype=np.float64)

    def learn_array(self, pattern: np.ndarray | collections.abc.Sequence[float]) -> GNG:
        """Update the GNG topology with a 1D numeric vector."""
        self.n_samples_seen += 1
        vec = np.array(pattern, dtype=np.float64)
        if len(self._units) < 2:
            self._units.append(self._new_unit(vec))
            return self

        d0 = self._dist(self._units[0].w, vec)
        d1 = self._dist(self._units[1].w, vec)
        if d0 <= d1:
            first, second = self._units[0], self._units[1]
            d_first, d_second = d0, d1
        else:
            first, second = self._units[1], self._units[0]
            d_first, d_second = d1, d0

        for j in range(2, len(self._units)):
            dj = self._dist(self._units[j].w, vec)
            if dj < d_first:
                second = first
                d_second = d_first
                first = self._units[j]
                d_first = dj
            elif dj < d_second:
                second = self._units[j]
                d_second = dj

        s1, s2 = first, second

        # 1. Increment age of all edges emanating from s1
        s1.update_ages()

        # 2. Accumulate local error in s1
        s1.error += d_first

        # 3. Move s1 and its direct topological neighbors
        s1.w += self.eps_b * (vec - s1.w)
        for neighbor in s1.get_neighborhood():
            neighbor.w += self.eps_n * (vec - neighbor.w)

        # 4. Connect s1 and s2 (or reset age to 0)
        edge = s1.search_edge(s2)
        if edge is None:
            s1.add_neighbor(s2)
        else:
            edge.age = 0

        # 5. Purge edges on s1 reaching max_age
        s1.purge_edges(self.max_age)

        # 6. Remove isolated units with no edges
        self._units = [u for u in self._units if u.neighbors]

        # 7. Insert a new unit every lambda_ samples
        if (
            self.n_samples_seen % self.lambda_ == 0
            and self._units
            and (self.max_nodes is None or len(self._units) < self.max_nodes)
        ):
            q = max(self._units, key=lambda u: u.error)
            nbrs = q.get_neighborhood()
            if nbrs:
                f = max(nbrs, key=lambda n: n.error)
                r = self._new_unit(0.5 * (q.w + f.w))
                r.add_neighbor(q)
                r.add_neighbor(f)
                q.remove_neighbor(f)

                q.error *= self.alpha
                f.error *= self.alpha
                r.error = q.error
                self._units.append(r)
                self.n_prototypes_created += 1

        # 8. Decay all error variables
        for u in self._units:
            u.error *= self.d

        return self

    def learn_one(self, x: dict) -> None:
        self.learn_array(self._dict_to_array(x, allow_new_features=True))

    def predict_one(self, x: dict) -> int:
        if not self._units:
            return 0
        vec = self._dict_to_array(x, allow_new_features=False)
        best_id = self._units[0].id
        best_dist = self._dist(self._units[0].w, vec)
        for u in self._units[1:]:
            dist = self._dist(u.w, vec)
            if dist < best_dist:
                best_dist = dist
                best_id = u.id
        return best_id

    @property
    def n_clusters(self) -> int:
        return len(self._units)

    @property
    def prototypes_array(self) -> np.ndarray:
        """Dense 2D `numpy.ndarray` of current unit weight vectors."""
        if not self._units:
            d = len(self._features) if self._features is not None else 0
            return np.zeros((0, d), dtype=np.float64)
        return np.array([u.w for u in self._units], dtype=np.float64)

    @property
    def centers(self) -> dict[int, dict]:
        """Dictionary mapping unit IDs to their prototype positions as feature dicts."""
        if not self._units:
            return {}
        keys = self._features if self._features is not None else list(range(len(self._units[0].w)))
        return {u.id: {k: float(u.w[i]) for i, k in enumerate(keys)} for u in self._units}

    def __deepcopy__(self, memo):
        new_gng = GNG(
            eps_b=self.eps_b,
            eps_n=self.eps_n,
            max_age=self.max_age,
            lambda_=self.lambda_,
            alpha=self.alpha,
            d=self.d,
            max_nodes=self.max_nodes,
            dist_metric=self.dist_metric,
            inv_cov=self.inv_cov,
            distance_func=self.distance_func,
        )
        memo[id(self)] = new_gng
        new_gng._features = list(self._features) if self._features is not None else None
        new_gng._feature_set = set(self._feature_set)
        new_gng._next_id = self._next_id
        new_gng.n_samples_seen = self.n_samples_seen
        new_gng.n_prototypes_created = self.n_prototypes_created

        # 1. Clone units preserving order
        unit_map: dict[_Unit, _Unit] = {}
        for u in self._units:
            new_u = _Unit(u.id, u.w.copy())
            new_u.error = u.error
            unit_map[u] = new_u
            new_gng._units.append(new_u)

        # 2. Reconstruct undirected edges preserving ages and per-unit neighbor order
        edge_map: dict[_Edge, _Edge] = {}
        for u in self._units:
            for e in u.neighbors:
                if e not in edge_map:
                    new_p0 = unit_map.get(e.p0)
                    new_p1 = unit_map.get(e.p1)
                    if new_p0 is not None and new_p1 is not None:
                        edge_map[e] = _Edge(new_p0, new_p1, age=e.age)

        for u in self._units:
            new_u = unit_map[u]
            for e in u.neighbors:
                new_e = edge_map.get(e)
                if new_e is not None:
                    new_u.neighbors.append(new_e)

        return new_gng

    @classmethod
    def _unit_test_params(cls):
        yield {"lambda_": 5, "max_age": 50}
