import numpy as np


def make_logical(n_tiles=1):
    """
        Make a toy dataset with three labels that represent the logical functions: OR, XOR, AND
        (functions of the 2D input).
    """

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
    np.random.shuffle(pat2)

    Y = np.array(pat2[:, 0:L], dtype=float)
    X = np.array(pat2[:, L:E], dtype=float)

    return X, Y
