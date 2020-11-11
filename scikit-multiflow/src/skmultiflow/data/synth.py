import numpy as np
from skmultiflow.utils import check_random_state


def make_logical(n_tiles=1, random_state=None):
    """ Make a toy dataset with three labels that represent the logical functions: OR, XOR, AND
     (functions of the 2D input).

    Parameters
    ----------
    n_tiles: int
        Number of tiles to generate

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    """
    random_state = check_random_state(random_state)

    pat = np.array([
        # X  X  Y  Y  Y
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 0, 1, 1, 0],
        [1, 1, 1, 0, 1]
    ], dtype=int)

    N, E = pat.shape
    D = 2
    L = E - D

    pat2 = np.zeros((N, E))
    pat2[:, 0:L] = pat[:, D:E]
    pat2[:, L:E] = pat[:, 0:D]
    pat2 = np.tile(pat2, (n_tiles, 1))
    random_state.shuffle(pat2)

    Y = np.array(pat2[:, 0:L], dtype=float)
    X = np.array(pat2[:, L:E], dtype=float)

    return X, Y
