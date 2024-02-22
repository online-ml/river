import numpy as np


def hankel(
    X: np.ndarray,
    hn: int,
    cut_rollover: bool = True,
) -> np.ndarray:
    """Create a Hankel matrix from a given input array.

    Args:
        X (np.ndarray): The input array.
        hn (int): The number of columns in the Hankel matrix.
        cut_rollover (bool, optional): Whether to cut the rollover part of the Hankel matrix. Defaults to True.

    Returns:
        np.ndarray: The Hankel matrix.

    Example:
    >>> X = np.array([1., 2., 3., 4., 5.])
    >>> hankel(X, 3, cut_rollover=False)
    array([[1., 2., 3.],
           [2., 3., 4.],
           [3., 4., 5.],
           [4., 5., 1.],
           [5., 1., 2.]])
    >>> hankel(X, 3)
    array([[1., 2., 3.],
           [2., 3., 4.],
           [3., 4., 5.]])
    """
    hX = np.empty((X.shape[0], hn))
    for i in range(hn):
        hX[:, i] = X
        X = np.roll(X, -1)
    if cut_rollover:
        hX = hX[: -hn + 1]
    return hX
