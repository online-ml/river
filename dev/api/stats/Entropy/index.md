# Entropy

Running entropy.

## Parameters

- **fading\_factor**

  *Type* → `float`

  *Default* → `1`

  Fading factor.
- **eps**

  *Type* → `float`

  *Default* → `1e-08`

  Small value that will be added to the denominator to avoid division by zero.

## Attributes

- **entropy** (`float`)

  The running entropy.
- **n** (`int`)

  The current number of observations.
- **counter** (`collections.Counter`)

  Count the number of times the values have occurred

## Examples

```python
import math
import random
import numpy as np
from scipy.stats import entropy
from river import stats

def entropy_list(labels, base=None):
    value,counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)

SEED = 42 * 1337
random.seed(SEED)

entro = stats.Entropy(fading_factor=1)

list_animal = []
for animal, num_val in zip(['cat', 'dog', 'bird'],[301, 401, 601]):
    list_animal += [animal for i in range(num_val)]
random.shuffle(list_animal)

for animal in list_animal:
    entro.update(animal)

print(f'{entro.get():.6f}')
```

```text
1.058093
```

```python
print(f'{entropy_list(list_animal):.6f}')
```

```text
1.058093
```

## Methods

> [!NOTE]
>
> **get**
>
> Return the current value of the statistic.

> [!NOTE]
>
> **update**
>
> Update the called instance.
>
> **Parameters**
>
> - **x** — `Hashable`

## References

- [Sovdat, B., 2014. Updating Formulas and Algorithms for Computing Entropy and Gini Index from Time-Changing Data Streams. arXiv preprint arXiv:1403.6348.](<https://arxiv.org/pdf/1403.6348.pdf>)
