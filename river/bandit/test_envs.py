from __future__ import annotations

import gym.utils.env_checker
import pytest

from river import bandit


def _iter_envs():
    for env_name in gym.envs.registry:
        if env_name.startswith(bandit.envs.RIVER_NAMESPACE):
            yield gym.make(env_name)


@pytest.mark.parametrize(
    "env",
    [pytest.param(env, id=env.unwrapped.__class__.__name__) for env in _iter_envs()],
)
def test_gym_check_env(env):
    gym.utils.env_checker.check_env(env.unwrapped)
