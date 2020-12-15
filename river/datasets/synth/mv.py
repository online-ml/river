import random

from .. import base


class Mv(base.SyntheticDataset):
    """Mv artificial dataset.

    Artificial dataset composed of both nominal and numeric features, whose features
    present co-dependencies. Originally described in [^1].

    The features are generated using the following expressions:

    - $x_1$: uniformly distributed over `[-5, 5]`.

    - $x_2$: uniformly distributed over `[-15, -10]`.

    - $x_3$:

        * if $x_1 > 0$, $x_3 \\leftarrow$ `'green'`

        * else $x_3 \\leftarrow$ `'red'` with probability $0.4$ and $x_3 \\leftarrow$ `'brown'`
        with probability $0.6$.

    - $x_4$:

        * if $x_3 =$ `'green'`, $x_4 \\leftarrow x_1 + 2 x_2$

        * else $x_4 = \\frac{x_1}{2}$ with probability $0.3$ and $x_4 = \\frac{x_2}{2}$
        with probability $0.7$.

    - $x_5$: uniformly distributed over `[-1, 1]`.

    - $x_6 \\leftarrow x_4 \\times \\epsilon$, where $\\epsilon$ is uniformly distributed
    over `[0, 5]`.

    - $x_7$: `'yes'` with probability $0.3$, and `'no'` with probability $0.7$.

    - $x_8$: `'normal'` if $x_5 < 0.5$ else `'large'`.

    - $x_9$: uniformly distributed over `[100, 500]`.

    - $x_{10}$: uniformly distributed integer over the interval `[1000, 1200]`.

    The target value is generated using the following rules:

    - if $x_2 > 2$, $y \\leftarrow 35 - 0.5 x_4$

    - else if $-2 \\le x_4 \\le 2$, $y \\leftarrow 10 - 2 x_1$

    - else if $x_7 =$ `'yes'`, $y \\leftarrow 3 - \\frac{x_1}{x_4}$

    - else if $x_8 =$ `'normal'`, $y \\leftarrow x_6 + x_1$

    - else $y \\leftarrow \\frac{x_1}{2}$.

    Parameters
    ----------
    seed
        Random seed number used for reproducibility.

    Examples
    --------
    >>> from river import synth

    >>> dataset = synth.Mv(seed=42)

    >>> for x, y in dataset.take(5):
    ...     print(list(x.values()), y)
    [1.39, -14.87, 'green', -28.35, -0.44, -31.64, 'no', 'normal', 370.67, 1178.43] -30.25
    [-4.13, -12.89, 'red', -2.06, 0.01, -0.27, 'yes', 'normal', 359.95, 1108.98] 1.00
    [-2.79, -12.05, 'brown', -1.39, 0.61, -4.87, 'no', 'large', 162.19, 1191.44] 15.59
    [-1.63, -14.53, 'red', -7.26, 0.20, -29.33, 'no', 'normal', 314.49, 1194.62] -30.96
    [-1.21, -12.23, 'brown', -6.11, 0.72, -17.66, 'no', 'large', 118.32, 1045.57] -0.60


    References
    ----------
    [^1]: [Mv in Luís Torgo regression datasets](https://www.dcc.fc.up.pt/~ltorgo/Regression/mv.html)

    """

    def __init__(self, seed: int = None):
        super().__init__(task=base.REG, n_features=10)
        self.seed = seed

    def __iter__(self):

        rng = random.Random(self.seed)

        while True:
            x = {1: rng.uniform(-5, 5), 2: rng.uniform(-15, -10)}

            if x[1] > 0:
                x[3] = "green"
            else:
                x[3] = rng.choices(population=["red", "brown"], weights=[0.4, 0.6])[0]

            if x[3] == "green":
                x[4] = x[1] + 2 * x[2]
            else:
                choice = rng.choices(population=[True, False], weights=[0.3, 0.7])[0]

                if choice:
                    x[4] = x[1] / 2
                else:
                    x[4] = x[2] / 2

            x[5] = rng.uniform(-1, 1)

            epsilon = rng.uniform(0, 5)
            x[6] = x[4] * epsilon

            x[7] = rng.choices(population=["yes", "no"], weights=[0.3, 0.7])[0]

            x[8] = "normal" if x[5] < 0.5 else "large"

            x[9] = rng.uniform(100, 500)

            x[10] = rng.uniform(1000, 1200)

            if x[2] > 2:
                y = 35 - 0.5 * x[4]
            elif -2 <= x[4] <= 2:
                y = 10 - 2 * x[1]
            elif x[7] == "yes":
                y = 3 - (x[1] / x[4] if x[4] != 0 else 0)
            elif x[8] == "normal":
                y = x[6] + x[1]
            else:
                y = x[1] / 2

            yield x, y
