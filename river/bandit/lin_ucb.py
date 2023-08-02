from __future__ import annotations

import collections
import functools
import random

from river import bandit, linear_model


class LinUCBDisjoint(bandit.base.ContextualPolicy):
    """LinUCB, disjoint variant.

    Although it works, as of yet it is too slow to realistically be used in practice.

    The way this works is that each arm is assigned a `linear_model.BayesianLinearRegression`
    instance. This instance is updated every time the arm is pulled. The context is used as
    features for the regression. The reward is used as the target. The posterior distribution
    is used to compute the upper confidence bound. The arm with the highest upper confidence
    bound is pulled.

    Parameters
    ----------
    alpha
        Parameter used in each Bayesian linear regression.
    beta
        Parameter used in each Bayesian linear regression.
    smoothing
        Parameter used in each Bayesian linear regression.
    reward_obj
        The reward object used to measure the performance of each arm.
    burn_in
        The number of time steps during which each arm is pulled once.
    seed
        Random number generator seed for reproducibility.

    References
    ----------
    [^1]: [A Contextual-Bandit Approach to Personalized News Article Recommendation](https://arxiv.org/abs/1003.0146)
    [^2:] [Contextual Bandits Analysis of LinUCB Disjoint Algorithm with Dataset](https://kfoofw.github.io/contextual-bandits-linear-ucb-disjoint/)

    """

    _REQUIRES_UNIVARIATE_REWARD = True

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        smoothing: float | None = None,
        reward_obj=None,
        burn_in=0,
        seed: int | None = None,
    ):
        super().__init__(reward_obj, burn_in)
        self.alpha = alpha
        self.beta = beta
        self.smoothing = smoothing
        self._bayes_lin_regs: collections.defaultdict[
            bandit.base.ArmID, linear_model.BayesianLinearRegression
        ] = collections.defaultdict(
            functools.partial(
                linear_model.BayesianLinearRegression,
                alpha=self.alpha,
                beta=self.beta,
                smoothing=self.smoothing,
            )
        )
        self.seed = seed
        self._rng = random.Random(seed)

    def _pull(self, arm_ids, context):
        def get_upper_bound(dist):
            return dist.mu + dist.sigma

        upper_bounds = {
            arm_id: get_upper_bound(
                self._bayes_lin_regs[arm_id].predict_one(context, with_dist=True)
            )
            for arm_id in arm_ids
        }
        biggest_upper_bound = max(upper_bounds.values())
        candidates = [
            arm_id
            for arm_id, upper_bound in upper_bounds.items()
            if upper_bound == biggest_upper_bound
        ]
        return self._rng.choice(candidates) if len(candidates) > 1 else candidates[0]

    def update(self, arm_id, context, *reward_args, **reward_kwargs):
        """Rewrite update function"""
        super().update(arm_id, None, *reward_args, **reward_kwargs)
        reward = reward_args[0]
        self._bayes_lin_regs[arm_id].learn_one(x=context, y=reward)
        return self
