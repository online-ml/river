from __future__ import annotations


def argsort(dists):
    return sorted(range(len(dists)), key=dists.__getitem__)


def rem_duplicates(pool):
    seen = set()
    return [n for n in pool if not (n in seen or seen.add(n))]
