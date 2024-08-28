from __future__ import annotations

import heapq
import math
import random

from river import base


class Vertex(base.Base):
    _isolated: set[int] = set()

    def __init__(self, item, uuid: int) -> None:
        self.item = item
        self.uuid = uuid
        self.edges: dict[int, float] = {}
        self.r_edges: dict[int, float] = {}
        self.flags: set[int] = set()
        self.worst_edge: int | None = None

    def __eq__(self, other) -> bool:
        if not isinstance(other, Vertex):
            raise NotImplementedError

        return self.uuid == other.uuid

    def __lt__(self, other) -> bool:
        if not isinstance(other, Vertex):
            raise NotImplementedError

        return self.uuid < other.uuid

    def farewell(self, vertex_pool: list[Vertex]):
        for rn in list(self.r_edges):
            vertex_pool[rn].rem_edge(self)

        for n in list(self.edges):
            self.rem_edge(vertex_pool[n])

        Vertex._isolated.discard(self.uuid)

    def fill(self, neighbors: list[Vertex], dists: list[float]):
        for n, dist in zip(neighbors, dists):
            self.edges[n.uuid] = dist
            self.flags.add(n.uuid)
            n.r_edges[self.uuid] = dist

        # Neighbors are ordered by distance, so the last neighbor
        # is the farthest one
        self.worst_edge = n.uuid

    def add_edge(self, vertex: Vertex, dist):
        self.edges[vertex.uuid] = dist
        self.flags.add(vertex.uuid)
        vertex.r_edges[self.uuid] = dist

        if self.worst_edge is None or self.edges[self.worst_edge] < dist:
            self.worst_edge = vertex.uuid

    def rem_edge(self, vertex: Vertex):
        self.edges.pop(vertex.uuid)
        vertex.r_edges.pop(self.uuid)
        self.flags.discard(vertex.uuid)

        if self.has_neighbors():
            if vertex.uuid == self.worst_edge:
                self.worst_edge = max(self.edges, key=self.edges.__getitem__)
        else:
            self.worst_edge = None

            if not self.has_rneighbors():
                Vertex._isolated.add(self.uuid)

    def push_edge(
        self, node: Vertex, dist: float, max_edges: int, vertex_pool: list[Vertex]
    ) -> int:
        if self.is_neighbor(node) or node.uuid == self.uuid:
            return 0

        if len(self.edges) >= max_edges:
            if self.worst_edge is None or self.edges.get(self.worst_edge, math.inf) <= dist:
                return 0
            self.rem_edge(vertex_pool[self.worst_edge])

        self.add_edge(node, dist)

        return 1

    def is_neighbor(self, vertex: Vertex):
        return vertex.uuid in self.edges or vertex.uuid in self.r_edges

    def get_edge(self, vertex: Vertex):
        if vertex.uuid in self.edges:
            return self, vertex, self.edges[vertex.uuid]
        return vertex, self, self.r_edges[vertex.uuid]

    def has_neighbors(self) -> bool:
        return len(self.edges) > 0

    def has_rneighbors(self) -> bool:
        return len(self.r_edges) > 0

    @property
    def sample_flags(self):
        return list(map(lambda n: n in self.flags, self.edges.keys()))

    @sample_flags.setter
    def sample_flags(self, sampled):
        self.flags -= set(sampled)

    def neighbors(self) -> tuple[list[int], list[float]]:
        res = tuple(map(list, zip(*((node, dist) for node, dist in self.edges.items()))))
        return res if len(res) > 0 else ([], [])  # type: ignore

    def r_neighbors(self) -> tuple[list[int], list[float]]:
        res = tuple(map(list, zip(*((vertex, dist) for vertex, dist in self.r_edges.items()))))
        return res if len(res) > 0 else ([], [])  # type: ignore

    def all_neighbors(self) -> set[int]:
        return set.union(set(self.edges.keys()), set(self.r_edges.keys()))

    def is_isolated(self):
        return len(self.edges) == 0 and len(self.r_edges) == 0

    def prune(
        self, prune_prob: float, prune_trigger: int, vertex_pool: list[Vertex], rng: random.Random
    ):
        if prune_prob == 0:
            return

        total_degree = len(self.edges) + len(self.r_edges)
        if total_degree <= prune_trigger:
            return

        edge_pool: list[tuple[float, int, bool]] = []
        for n, dist in self.edges.items():
            heapq.heappush(edge_pool, (dist, n, True))

        for rn, dist in self.r_edges.items():
            heapq.heappush(edge_pool, (dist, rn, False))

        # Start with the best undirected edge
        selected: list[int] = [heapq.heappop(edge_pool)[1]]
        while len(edge_pool) > 0:
            c_dist, c, c_isdir = heapq.heappop(edge_pool)
            discarded = False
            for s in selected:
                s_v = vertex_pool[s]
                c_v = vertex_pool[c]
                if s_v.is_neighbor(c_v) and rng.random() < prune_prob:
                    orig, dest, dist = s_v.get_edge(c_v)
                    if dist < c_dist:
                        if c_isdir:
                            self.rem_edge(c_v)
                        else:
                            c_v.rem_edge(self)
                        discarded = True
                        break
                    else:
                        orig.rem_edge(dest)

            if not discarded:
                selected.append(c)
