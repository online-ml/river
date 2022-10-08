import gym
from .candy_cane import CandyCaneContest

__all__ = ["CandyCaneContest"]

RIVER_NAMESPACE = 'river_bandits'

if (env_id := f'{RIVER_NAMESPACE}/CandyCaneContest-v0') not in gym.envs.registry:
    gym.envs.registration.register(
        id=env_id,
        entry_point='river.bandit.envs:CandyCaneContest',
        max_episode_steps=CandyCaneContest.n_rounds,
    )
