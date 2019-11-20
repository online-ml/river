import copy


class SuccessiveHalving:
    """Successive halving with an infinite horizon.

    References:
        1. `Non-stochastic Best Arm Identification and Hyperparameter Optimization <http://proceedings.mlr.press/v51/jamieson16.pdf>`_
        2. `Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization <https://homes.cs.washington.edu/~jamieson/hyperband.html>`_
        3. `Massively Parallel Hyperparameter Tuning <https://arxiv.org/pdf/1810.05934.pdf>`_

    """

    def __init__(self, model, param_grid, metric):
        self.S = [len(param_grid)]
        self.models = [
            copy.deepcopy(model)._set_params(params)
            for params in param_grid
        ]
        self.metrics = [copy.copy(metric) for _ in range(len(param_grid))]


class Hyperband:
    """Hyperband with an infinite horizon.

    References:
        1. `Embracing the Random <http://www.argmin.net/2016/06/23/hyperband/>`_
        2. `Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization <https://homes.cs.washington.edu/~jamieson/hyperband.html>`_

    """
