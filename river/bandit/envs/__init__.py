from __future__ import annotations

try:
    import gym

    GYM_INSTALLED = True
except ImportError:
    GYM_INSTALLED = False

if GYM_INSTALLED:
    from .candy_cane import CandyCaneContest
    from .testbed import KArmedTestbed

    __all__ = ["CandyCaneContest", "KArmedTestbed"]

    RIVER_NAMESPACE = "river_bandits"

    if (env_id := f"{RIVER_NAMESPACE}/CandyCaneContest-v0") not in gym.envs.registry:
        gym.envs.registration.register(
            id=env_id,
            entry_point="river.bandit.envs:CandyCaneContest",
            max_episode_steps=CandyCaneContest.n_steps,
        )
    if (env_id := f"{RIVER_NAMESPACE}/KArmedTestbed-v0") not in gym.envs.registry:
        gym.envs.registration.register(
            id=env_id,
            entry_point="river.bandit.envs:KArmedTestbed",
            max_episode_steps=KArmedTestbed.n_steps,
        )
