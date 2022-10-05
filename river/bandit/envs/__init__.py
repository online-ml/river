import gym
from .candy_cane import CandyCaneContest

__all__ = ["CandyCaneContest"]

if (env_id := 'river_bandits/CandyCaneContest-v0') not in gym.envs.registry:
    gym.envs.registration.register(
        id=env_id,
        entry_point='river.bandit.envs:CandyCaneContest',
        max_episode_steps=CandyCaneContest.n_rounds,
    )
