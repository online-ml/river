from __future__ import annotations

import collections
import functools
import heapq
import itertools
import math
import operator
import random
import typing

from river import utils
from river.neighbors.base import BaseNN, DistanceFunc, FunctionWrapper

from .nn_vertex import Vertex


class SWINN(BaseNN):
    """Sliding WIndow-based Nearest Neighbor (SWINN) search using Graphs.

    Extends the NNDescent algorithm[^1] to handle vertex addition and removal in a FIFO data
    ingestion policy. SWINN builds and keeps a directed graph where edges connect the nearest
    neighbors. Any distance metric can be used to build the graph. By using a directed graph,
    the user must set the desired number of neighbors. More neighbors imply more accurate
    search queries at the cost of increased running time and memory usage. Note that although
    the number of directed neighbors is limited by the user, there is no direct control on the
    number of reverse neighbors, i.e., the number of vertices that have an edge to a given vertex.

    The basic idea of SWINN and NNDescent is that "the neighbor of my neighbors might as well be
    my neighbor". Hence, the connections are constantly revisited to improve the graph structure.
    The algorithm for creating and maintaining the search graph can be described
    in general lines as follows:

    * Start with a random neighborhood graph;

    * For each node in the search graph: refine the current neighborhood by checking if there
    are better neighborhood options among the neighbors of the current neighbors;

    * If the total number of neighborhood changes is smaller than a given stopping criterion, then stop.

    SWINN adds strategies to remove vertices from the search graph and pruning redundant edges. SWINN is
    more efficient when the selected `maxlen` is greater than 500. For small sized data windows, using
    the lazy/exhaustive search, i.e., `neighbors.LazySearch` might be a better idea.

    Parameters
    ----------
    graph_k
        The maximum number of direct nearest neighbors each node has.
    maxlen
        The maximum size of the data buffer.
    warm_up
        How many data instances to observe before starting the search graph.
    dist_func
        The distance function used to compare two items. If not set, use the Minkowski distance
        with `p=2`.
    max_candidates
        The maximum number of vertices to consider when performing local neighborhood joins. If not set
        SWINN will use `min(50, max(50, self.graph_k))`.
    delta
        Early stop parameter for the neighborhood refinement procedure. NNDescent will stop running
        if the maximum number of iterations is reached or the number of edge changes after an iteration
        is smaller than or equal to `delta * graph_k * n_nodes`. In the last expression, `n_nodes`
        refers to the number of graph nodes involved in the (local) neighborhood refinement.
    prune_prob
        The probability of removing redundant edges. Must be between `0` and `1`. If set to zero,
        no edge will be pruned. When set to one, every potentially redundant edge will be dropped.
    n_iters
        The maximum number of NNDescent iterations to perform to refine the search index.
    seed
        Random seed for reproducibility.

    Notes
    -----
    There is an accuracy/speed trade-off between `graph_k` and `sample_rate`. To ensure a single
    connected component, and thus an effective search index, one can increase `graph_k`. The
    `connectivity` method is a helper to determine whether the search index has a single connected component.
    However, search accuracy might come at the cost of increased memory usage and slow processing. To alleviate
    that, one can rely on decreasing the `sample_rate` to avoid exploring all the undirected edges of a node
    during search queries and local graph refinements. Moreover, the edge pruning procedures also help
    decreasing the computational costs. Note that, anything that limits the number of explored neighbors or
    prunes edges might have a negative impact on search accuracy.

    References
    ----------
    [^1]: Dong, W., Moses, C., & Li, K. (2011, March). Efficient k-nearest neighbor graph construction for
    generic similarity measures. In Proceedings of the 20th international conference on World wide web (pp. 577-586).

    """

    def __init__(
        self,
        graph_k: int = 20,
        dist_func: DistanceFunc | FunctionWrapper | None = None,
        maxlen: int = 1000,
        warm_up: int = 500,
        max_candidates: int = None,
        delta: float = 0.0001,
        prune_prob: float = 0.0,
        n_iters: int = 10,
        seed: int = None,
    ):
        self.graph_k = graph_k
        if dist_func is None:
            dist_func = functools.partial(utils.math.minkowski_distance, p=2)
        self.dist_func = dist_func

        self.maxlen = maxlen
        self.warm_up = warm_up
        if max_candidates is None:
            self.max_candidates = min(50, max(50, self.graph_k))
        else:
            self.max_candidates = max_candidates

        self.delta = delta
        self.prune_prob = prune_prob

        self.n_iters = n_iters
        self.seed = seed

        self._data: collections.deque[Vertex] = collections.deque(maxlen=self.maxlen)
        self._uuid = itertools.cycle(range(self.maxlen))
        self._rng = random.Random(self.seed)
        self._index = False

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i]

    def __iter__(self):
        yield from self._data

    def _init_graph(self):
        """Create a random nearest neighbor graph."""
        n_nodes = len(self)

        nodes = set([i for i in range(n_nodes)])
        for nid in range(n_nodes):
            nodes.remove(nid)
            ns = self._rng.sample(tuple(nodes), self.graph_k)
            dists = [math.inf for _ in range(self.graph_k)]
            self[nid].fill([self[n] for n in ns], dists)
            nodes.add(nid)

    def _fix_graph(self):
        """Connect every isolated node in the graph to their nearest neighbors."""

        for node in list(Vertex._isolated):
            if not node.is_isolated():
                continue
            neighbors, dists = self._search(node.item, self.graph_k)
            node.fill(neighbors, dists)

        # Update class property
        Vertex._isolated.clear()

    def _safe_node_removal(self):
        """Remove the oldest data point from the search graph.

        Make sure nodes are accessible from any given starting point after removing the oldest
        node in the search graph. New traversal paths will be added in case the removed node was
        the only bridge between its neighbors.

        """
        node = self._data.popleft()
        # Get previous neighborhood info
        rns = node.r_neighbors()[0]
        ns = node.neighbors()[0]
        node.farewell()

        # Nodes whose only direct neighbor was the removed node
        rns = {rn for rn in rns if not rn.has_neighbors()}
        # Nodes whose only reverse neighbor was the removed node
        ns = {n for n in ns if not n.has_rneighbors()}

        affected = list(rns | ns)
        isolated = rns.intersection(ns)

        # First we handle the unreachable nodes
        for al in isolated:
            neighbors, dists = self._search(al.item, self.graph_k)
            al.fill(neighbors, dists)

        rns -= isolated
        ns -= isolated
        ns = tuple(ns)

        # Nodes with no direct neighbors
        for rn in rns:
            seed = None
            # Check the group of nodes without reverse neighborhood for seeds
            # Thus we can join two separate groups
            if len(ns) > 0:
                seed = self._rng.choice(ns)

            # Use the search index to create new connections
            neighbors, dists = self._search(rn.item, self.graph_k, seed=seed, exclude=rn)
            rn.fill(neighbors, dists)

        self._refine(affected)

    def _refine(self, nodes: list[Vertex] = None):
        """Update the nearest neighbor graph to improve the edge distances.

        Parameters
        ----------
        nodes
            The list of nodes for which the neighborhood refinement will be applied.
            If `None`, all nodes will have their neighborhood enhanced.
        """

        if nodes is None:
            nodes = [n for n in self]

        min_changes = self.delta * self.graph_k * len(nodes)

        tried = set()
        for _ in range(self.n_iters):
            total_changes = 0

            new = collections.defaultdict(set)
            old = collections.defaultdict(set)

            # Expand undirected neighborhood
            for node in nodes:
                neighbors = node.neighbors()[0]
                flags = node.sample_flags

                for neigh, flag in zip(neighbors, flags):
                    # To avoid evaluating previous neighbors again
                    tried.add((node.uuid, neigh.uuid))
                    if flag:
                        new[node].add(neigh)
                        new[neigh].add(node)
                    else:
                        old[node].add(neigh)
                        old[neigh].add(node)

            # Limits the maximum number of edges to explore and update sample flags
            for node in nodes:
                if len(new[node]) > self.max_candidates:
                    new[node] = self._rng.sample(tuple(new[node]), self.max_candidates)  # type: ignore
                else:
                    new[node] = new[node]

                if len(old[node]) > self.max_candidates:
                    old[node] = self._rng.sample(tuple(old[node]), self.max_candidates)  # type: ignore
                else:
                    old[node] = old[node]

                node.sample_flags = new[node]

            # Perform local joins an attempt to improve the neighborhood
            for node in nodes:
                # The origin of the join must have a boolean flag set to true
                for n1 in new[node]:
                    # Consider connections between vertices whose boolean flags are both true
                    for n2 in new[node]:
                        if n1.uuid == n2.uuid or n1.is_neighbor(n2):
                            continue

                        if (n1.uuid, n2.uuid) in tried or (n2.uuid, n1.uuid) in tried:
                            continue

                        dist = self.dist_func(n1.item, n2.item)
                        total_changes += n1.push_edge(n2, dist, self.graph_k)
                        total_changes += n2.push_edge(n1, dist, self.graph_k)

                        tried.add((n1.uuid, n2.uuid))

                    # Or one of the connections has a boolean flag set to false
                    for n2 in old[node]:
                        if n1.uuid == n2.uuid or n1.is_neighbor(n2):
                            continue

                        if (n1.uuid, n2.uuid) in tried or (n2.uuid, n1.uuid) in tried:
                            continue

                        dist = self.dist_func(n1.item, n2.item)
                        total_changes += n1.push_edge(n2, dist, self.graph_k)
                        total_changes += n2.push_edge(n1, dist, self.graph_k)

                        tried.add((n1.uuid, n2.uuid))

            # Stopping criterion
            if total_changes <= min_changes:
                break

        # Reduce the number of edges, if needed
        for n in nodes:
            n.prune(self.prune_prob, self.max_candidates, self._rng)

        # Ensure that no node is isolated in the graph
        self._fix_graph()

    def append(self, item: typing.Any, **kwargs):
        """Add a new item to the search index.

        Data is stored using the FIFO strategy. Both the data buffer and the search graph are updated. The
        addition of a new item will trigger the removal of the oldest item, if the maximum size was
        reached. All edges of the removed node are also dropped and safety procedures are applied to ensure
        its neighbors keep accessible. The addition of a new item also trigger local neighborhood refinement
        procedures, to ensure the search index is effective and the node degree constraints are met.

        Parameters
        ----------
        item
            Item to be added.
        kwargs
            Not used in this implementation.

        """
        node = Vertex(item, next(self._uuid))
        if not self._index:
            self._data.append(node)
            if len(self) >= self.warm_up:
                self._init_graph()
                self._refine()
                self._index = True
            return

        # A slot will be replaced, so let's update the search graph first
        if len(self) == self.maxlen:
            self._safe_node_removal()

        # Assign the closest neighbors to the new item
        neighbors, dists = self._search(node.item, self.graph_k)

        # Add the new element to the buffer
        self._data.append(node)
        node.fill(neighbors, dists)

    def _linear_scan(self, item, k):
        # Lazy search while the warm-up period is not finished
        points = [(p.item, self.dist_func(item, p.item)) for p in self]

        if points:
            return tuple(map(list, zip(*sorted(points, key=operator.itemgetter(-1))[:k])))

        return None

    def _search(self, item, k, epsilon: float = 0.1, seed=None, exclude=None) -> tuple[list, list]:
        # Limiter for the distance bound
        distance_scale = 1 + epsilon
        # Distance threshold for early stops
        distance_bound = math.inf

        if exclude is None:
            exclude = set()
        else:
            exclude = {exclude.uuid}

        if seed is None:
            # Make sure the starting point for the search is valid
            while True:
                # Random seed point to start the search
                seed = self[self._rng.randint(0, len(self) - 1)]
                if not seed.is_isolated() and seed.uuid not in exclude:
                    break

        dist = self.dist_func(item, seed.item)

        # To avoid computing distances more than once for a given node
        visited = {seed.uuid}
        visited |= exclude

        # Search pool is a minimum heap
        pool = [(dist, seed)]

        # Results are stored in a maximum heap
        result = [(-dist, seed)]

        c_dist, c_n = heapq.heappop(pool)
        while c_dist < distance_bound:
            tns = [n for n in c_n.all_neighbors() if n.uuid not in visited]

            for n in tns:
                dist = self.dist_func(item, n.item)

                if len(result) < k:
                    heapq.heappush(result, (-dist, n))
                    heapq.heappush(pool, (dist, n))
                    distance_bound = distance_scale * -result[0][0]
                elif dist < -result[0][0]:
                    heapq.heapreplace(result, (-dist, n))
                    heapq.heappush(pool, (dist, n))
                    distance_bound = distance_scale * -result[0][0]
                visited.add(n.uuid)
            if len(pool) == 0:
                break
            c_dist, c_n = heapq.heappop(pool)

        result.sort(reverse=True)
        neighbors, dists = map(list, zip(*((r[1], -r[0]) for r in result)))

        return neighbors, dists

    def search(
        self, item: typing.Any, n_neighbors: int, epsilon: float = 0.1, **kwargs
    ) -> tuple[list, list]:
        """Search the underlying nearest neighbor graph given a query item.

        In case not enough samples were observed, i.e., the number of stored samples is smaller than
        `warm_up`, then the search switches to a brute force strategy.

        Parameters
        ----------
        item
            The query item to search for nearest neighbors.
        n_neighbors
            The number of nearest neighbors to return.
        epsilon
            Distance bound to aid in avoiding local minima while traversing the search graph. Let $d_k$
            be the distance of the query item to current $k$-th nearest neighbor. At any given
            moment, any point whose distance to the query item is smaller than or equal to
            $(1 + \\epsilon) * d_k$ is kept as a potential path. After every addition to the heap of
            candidate nodes, $d_k$ is updated.
        kwargs
            Not used in this implementation.

        Returns
        -------
        neighbors, dists
            A tuple containing the id of the neighbors in the buffer and the respective distances to them.

        """

        if len(self) <= self.warm_up:
            return self._linear_scan(item, n_neighbors)

        neighbors, dists = self._search(item, n_neighbors, epsilon)
        return [n.item for n in neighbors], dists

    def connectivity(self) -> list[int]:
        """Get a list with the size of each connected component in the search graph.

        This metric provides an overview of reachability in the search index by using Kruskal's
        algorithm to build a forest of connected components.

        We want our search index to have a single connected component, i.e., the case where we get
        a list containing a single number which is equal to `maxlen`. If that is not the case, not
        every node in the search graph can be reached from any given starting point. You may want to try
        increasing `graph_k` to improve connectivity. However, keep in mind the following aspects:
        1) computing this metric is a costly operation ($O(E\\log V)$), where $E$ and $V$ are, respectively,
        the number of edges and vertices in the search graph; 2) often, connectivity comes at the price of
        increased computational costs. Tweaking the `sample_rate` might help in such situations. The best
        possible scenario is to decrease the value of `graph_k` while keeping a single connected
        component.

        Returns
        -------
        A list of the number of elements in each connected component of the graph.

        """
        forest = set()
        trees = {n: {n} for n in self}

        edges = [((n1, n2), w) for n1 in self for n2, w in n1.edges.items()]
        edges.sort(key=operator.itemgetter(1))

        for (n1, n2), _ in edges:
            if trees[n1].isdisjoint(trees[n2]):
                forest.discard(frozenset(trees[n1]))
                forest.discard(frozenset(trees[n2]))

                u = trees[n1] | trees[n2]
                # Update the trees
                for v in u:
                    trees[v] = u

                forest.add(frozenset(u))

        return [len(tree) for tree in forest]
