# NewsArticles

News articles bandit dataset.

This is a personalization dataset. It contains 10000 observations. There are 10 arms, and the reward is binary. There are 100 features, which turns this into a contextual bandit problem.

## Attributes

- **arms**

  The list of arms that can be pulled.
- **desc**

  Return the description from the docstring.
- **is\_downloaded**

  Indicate whether or the data has been correctly downloaded.
- **path**

## Examples

```python
from river import bandit

dataset = bandit.datasets.NewsArticles()
context, arm, reward = next(iter(dataset))

len(context)
```

```text
100
```

```python
arm, reward
```

```text
(2, False)
```

## Methods

> [!NOTE]
>
> **download**

> [!NOTE]
>
> **take**
>
> Iterate over the k samples.
>
> **Parameters**
>
> - **k** — `int`

## References

- [Machine Learning for Personalization homework](<http://www.cs.columbia.edu/~jebara/6998/hw2.pdf>)
- [Contextual Bandits Analysis of LinUCB Disjoint Algorithm with Dataset](<https://kfoofw.github.io/contextual-bandits-linear-ucb-disjoint/>)
