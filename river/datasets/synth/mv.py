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

        * if $x_3 = $ `'green'`, $x_4 \\leftarrow x_1 + 2x_2$

        * else $x_4 = \\frac{x_1}{2}$ with probability $0.3$ and $x_4 = \\frac{x_2}{2}$
        with probability $0.7$.

    - $x_5$: uniformly distributed over `[-1, 1]`.

    - $x_6$: $x_6 \\leftarrow x_4 \\times \\epsilon$, where $\\epsilon$ is uniformly distributed
    over `[0, 5]`.

    - $x_7$: `'yes'` with probability $0.3$, and `'no'` with probability $0.7$.

    - $x_8$: `normal` if $x_5 < 0.5$ else `'large'`.

    - $x_9$: uniformly distributed over `[100, 500]`.

    - $x_{10}$: uniformly distributed integer over the interval `[1000, 1200]`.

    The target value is generated using the following rules:

    - if $x_2 > 2$, $y \\leftarrow 35 - 0.5 x_4$

    - else if $-2 \\le x_4 \\le 2$, $y \\leftarrow 10 - 2 x_1$

    - else if $x_7 =$ `'yes'`, $y \\leftarrow 3 - \frac{x_1}{x_4}$

    - else if $x_8 =$ `'normal'`, $y \\leftarrow x_6 + x_1$

    - else $y \\leftarrow \\frac{x_1}{2}$.

    Parameters
    ----------
    seed
        Random seed number used for reproducibility.

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
            pass
