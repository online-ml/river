from __future__ import annotations

import copy
import importlib
import inspect
import random

import gym
import pytest

from river import bandit, metrics

from .test_envs import _iter_envs


def test_ranking():
    class DummyMetric(metrics.base.Metric):
        def __init__(self):
            self.value = None

        def get(self):
            return self.value

        @property
        def bigger_is_better(self):
            return False

        def revert(self):
            ...

        def update(self):
            ...

        def works_with(self):
            ...

    class DummyPolicy(bandit.base.Policy):
        def __init__(self):
            super().__init__(reward_obj=DummyMetric())

        def _pull(self, arms):
            ...

    policy = DummyPolicy()
    policy._rewards[0].value = 0
    policy._rewards[1].value = 1
    policy._rewards[2].value = 2
    assert policy.ranking == [0, 1, 2]

    policy._rewards[0].value = 2
    policy._rewards[1].value = 1
    policy._rewards[2].value = 0
    assert policy.ranking == [2, 1, 0]

    policy._rewards[0].value = 0
    policy._rewards[1].value = 2
    policy._rewards[2].value = 1
    assert policy.ranking == [0, 2, 1]


class RandomPolicy(bandit.base.Policy):
    def _pull(self, arm_ids):
        return random.choice(arm_ids)


def _iter_policies():
    for _, policy in inspect.getmembers(importlib.import_module("river.bandit"), inspect.isclass):
        for params in policy._unit_test_params():
            yield policy(**params)


@pytest.mark.parametrize(
    "policy,env",
    [
        pytest.param(
            policy, env, id=f"{policy.__class__.__name__}-{env.unwrapped.__class__.__name__}"
        )
        # TODO: add simpler environments to test with
        for policy in _iter_policies()
        for env in _iter_envs()
    ],
)
@pytest.mark.skip(reason="flaky")
def test_better_than_random_policy(policy: bandit.base.Policy, env: gym.Env):
    """Test that the policy is better than random."""

    n_trials = 30
    n_successes = 0

    for _ in range(n_trials):
        policy = policy.clone()
        random_policy = RandomPolicy()

        env_seed = random.randint(0, 2**32 - 1)
        env = copy.deepcopy(env)
        _ = env.reset(seed=env_seed)
        _ = env.action_space.seed(env_seed)

        random_env = copy.deepcopy(env)
        _ = random_env.reset(seed=env_seed)
        _ = random_env.action_space.seed(env_seed)

        policy_reward = 0.0
        random_reward = 0.0

        terminated, truncated = False, False

        arm_ids = list(range(env.action_space.n))  # type: ignore

        while not terminated and not truncated:
            arm_id = policy.pull(arm_ids)  # type: ignore
            observation, reward, terminated, truncated, info = env.step(arm_id)
            policy.update(arm_id, reward)
            policy_reward += reward

            random_arm_id = random_policy.pull(arm_ids)  # type: ignore
            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = random_env.step(random_arm_id)
            random_policy.update(random_arm_id, reward)
            random_reward += reward

        n_successes += policy_reward > random_reward

    assert n_successes > n_trials * 0.8
