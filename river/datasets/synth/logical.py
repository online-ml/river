from __future__ import annotations

import itertools

import numpy as np

from river import datasets


class Logical(datasets.base.SyntheticDataset):
    """Logical functions stream generator.

    Make a toy dataset with three labels that represent the logical
    functions: `OR`, `XOR`, `AND` (functions of the 2D input).

    Data is generated in 'tiles' which contain the complete set of
    logical operations results. The tiles are repeated `n_tiles` times.
    Optionally, the generated data can be shuffled.

    Parameters
    ----------
    n_tiles
        Number of tiles to generate.
    shuffle
        If set, generated data will be shuffled.
    seed
        If int, `seed` is used to seed the random number generator;
        If RandomState instance, `seed` is the random number generator;
        If None, the random number generator is the `RandomState` instance used
        by `np.random`.

    Examples
    --------
    >>> from river.datasets import synth

    >>> dataset = synth.Logical(n_tiles=2, shuffle=True, seed=42)

    >>> for x, y in dataset.take(5):
    ...     print(x, y)
    {'A': 0, 'B': 1} {'OR': 1, 'XOR': 1, 'AND': 0}
    {'A': 0, 'B': 1} {'OR': 1, 'XOR': 1, 'AND': 0}
    {'A': 0, 'B': 0} {'OR': 0, 'XOR': 0, 'AND': 0}
    {'A': 1, 'B': 1} {'OR': 1, 'XOR': 0, 'AND': 1}
    {'A': 1, 'B': 0} {'OR': 1, 'XOR': 1, 'AND': 0}

    """

    def __init__(
        self,
        n_tiles: int = 1,
        shuffle: bool = True,
        seed: int | np.random.RandomState | None = None,
    ):
        super().__init__(
            n_features=2,
            n_outputs=3,
            n_samples=4 * n_tiles,
            task=datasets.base.MO_BINARY_CLF,
        )
        self.n_tiles = n_tiles
        self.shuffle = shuffle
        self.seed = seed
        self.rng = seed if isinstance(seed, np.random.RandomState) else np.random.RandomState(seed)
        self.feature_names = ["A", "B"]
        self.target_names = ["OR", "XOR", "AND"]

    def __iter__(self):
        X, Y = self._make_logical(n_tiles=self.n_tiles, shuffle=self.shuffle)

        for xi, yi in itertools.zip_longest(X, Y if hasattr(Y, "__iter__") else []):
            yield dict(zip(self.feature_names, xi)), dict(zip(self.target_names, yi))

    def _make_logical(self, n_tiles: int = 1, shuffle: bool = True):
        """Make toy dataset"""
        base_pattern = np.array(
            [
                # A  B  OR  XOR  AND
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [1, 0, 1, 1, 0],
                [1, 1, 1, 0, 1],
            ],
            dtype=int,
        )

        N, E = base_pattern.shape
        D = 2
        L = E - D

        pattern = np.zeros((N, E))
        pattern[:, 0:L] = base_pattern[:, D:E]
        pattern[:, L:E] = base_pattern[:, 0:D]
        pattern = np.tile(pattern, (n_tiles, 1))
        if shuffle:
            self.rng.shuffle(pattern)
        # return X, Y
        return (
            np.array(pattern[:, L:E], dtype=int),
            np.array(pattern[:, 0:L], dtype=int),
        )
