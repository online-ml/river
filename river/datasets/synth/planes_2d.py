import random

from river import datasets


class Planes2D(datasets.base.SyntheticDataset):
    """2D Planes synthetic dataset.

    This dataset is described in [^1] and was adapted from [^2]. The features are generated
    using the following probabilities:

    $$P(x_1 = -1) = P(x_1 = 1) = \\frac{1}{2}$$

    $$P(x_m = -1) = P(x_m = 0) = P(x_m = 1) = \\frac{1}{3}, m=2,\\ldots, 10$$

    The target value is defined by the following rule:

    $$\\text{if}~x_1 = 1, y \\leftarrow 3 + 3x_2 + 2x_3 + x_4 + \\epsilon$$

    $$\\text{if}~x_1 = -1, y \\leftarrow -3 + 3x_5 + 2x_6 + x_7 + \\epsilon$$

    In the expressions, $\\epsilon \\sim \\mathcal{N}(0, 1)$, is the noise.

    Parameters
    ----------
    seed
        Random seed number used for reproducibility.

    Examples
    --------
    >>> from river.datasets import synth

    >>> dataset = synth.Planes2D(seed=42)

    >>> for x, y in dataset.take(5):
    ...     print(list(x.values()), y)
    [-1, -1, 1, 0, -1, -1, -1, 1, -1, 1] -9.07
    [1, -1, -1, -1, -1, -1, 1, 1, -1, 1] -4.25
    [-1, 1, 1, 1, 1, 0, -1, 0, 1, 0] -0.95
    [-1, 1, 0, 0, 0, -1, -1, 0, -1, -1] -6.10
    [1, -1, 0, 0, 1, 0, -1, 1, 0, 1] 1.60

    References
    ----------
    [^1]: [2DPlanes in Luís Torgo regression datasets](https://www.dcc.fc.up.pt/~ltorgo/Regression/2dplanes.html)
    [^2]: Breiman, L., Friedman, J., Stone, C.J. and Olshen, R.A., 1984. Classification and
    regression trees. CRC press.

    """

    def __init__(self, seed: int = None):
        super().__init__(task=datasets.base.REG, n_features=10)
        self.seed = seed

    def __iter__(self):

        rng = random.Random(self.seed)

        while True:
            x = {1: rng.choice([-1, 1])}

            for m in range(2, 11):
                x[m] = rng.randint(-1, 1)

            if x[1] == 1:
                y = 3 + 3 * x[2] + 2 * x[3] + x[4]
            else:
                y = -3 + 3 * x[5] + 2 * x[6] + x[7]

            # Add noise
            y += rng.gauss(mu=0, sigma=1)

            yield x, y
