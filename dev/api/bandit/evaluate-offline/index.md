# evaluate\_offline

Evaluate a policy on historical logs using replay.

This is a high-level utility function for evaluating a policy using the replay methodology. This methodology is an off-policy evaluation method. It does not require an environment, and is instead data-driven.

At each step, an arm is pulled from the provided policy. If the arm is the same as the arm that was pulled in the historical data, the reward is used to update the policy. If the arm is different, the reward is ignored. This is the off-policy aspect of the evaluation.

## Parameters

- **policy**

  *Type* → `bandit.base.Policy`

  The policy to evaluate.
- **history**

  *Type* → `History | bandit.datasets.BanditDataset`

  The history of the bandit problem. This is a generator that yields tuples of the form `(arms, context, arm, reward)`.
- **reward\_stat**

  *Type* → `stats.base.Univariate | None`

  *Default* → `None`

  The reward statistic to use. Defaults to `stats.Sum`.

## Examples

```python
import random
from river import bandit

rng = random.Random(42)
arms = ['A', 'B', 'C']
clicks = [
    (
        arms,
        # no context
        None,
        # random arm
        rng.choice(arms),
        # reward
        rng.random() > 0.5
    )
    for _ in range(1000)
]

total_reward, n_samples_used = bandit.evaluate_offline(
    policy=bandit.EpsilonGreedy(0.1, seed=42),
    history=clicks,
)

total_reward
```

```text
Sum: 172.
```

```python
n_samples_used
```

```text
321
```

This also works out of the box with datasets that inherit from `river.bandit.BanditDataset`.

```python
news = bandit.datasets.NewsArticles()
total_reward, n_samples_used = bandit.evaluate_offline(
    policy=bandit.RandomPolicy(seed=42),
    history=news,
)

total_reward, n_samples_used
```

```text
(Sum: 105., 1027)
```

As expected, the policy succeeds in roughly 10% of cases. Indeed, there are 10 arms and 10000 samples, so the expected number of successes is 1000.

## References

- [Offline Evaluation of Multi-Armed Bandit Algorithms in Python using Replay](<https://jamesrledoux.com/algorithms/offline-bandit-evaluation/>)
- [Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms](<https://arxiv.org/pdf/1003.5956.pdf>)
- [Understanding Inverse Propensity Score for Contextual Bandits](<https://chanind.github.io/ai/2022/03/14/inverse-propensity-score-contextual-bandits.html>)
