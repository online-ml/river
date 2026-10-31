# HeavyHitters

Find the Heavy Hitters using the Lossy Count with Forgetting factor algorithm[^/dev/api/sketch/HeavyHitters/#1].

Keep track of the most frequent item(set)s in a data stream and apply a forgetting factor to discard previous frequent items that do not often appear anymore. This is an approximation algorithm designed to work with a limited amount of memory rather than accounting for every possible solution (thus using an unbounded memory footprint). Any hashable type can be passed as input, hence tuples or frozensets can also be monitored.

Considering a data stream where `n` elements were observed so far, the Lossy Count algorithm has the following properties:

- All item(set)s whose true frequency exceeds `support * n` are output. There are no

false negatives;

- No item(set) whose true frequency is less than `(support - epsilon) * n` is outputted;
- Estimated frequencies are less than the true frequencies by at most `epsilon * n`.

## Parameters

- **support**

  *Type* → `float`

  *Default* → `0.001`

  The support threshold used to determine if an item is frequent. The value of `support` must be in \\(\\\[0, 1\\\]\\). Elements whose frequency is higher than `support` times the number of observations seen so far are outputted.
- **epsilon**

  *Type* → `float`

  *Default* → `0.005`

  Error parameter to control the accuracy-memory tradeoff. The value of `epsilon` must be in \\((0, 1\]\\) and typically `epsilon` \\(\\ll\\) `support`. The smaller the `epsilon`, the more accurate the estimates will be, but the count sketch will have an increased memory footprint.
- **fading\_factor**

  *Type* → `float`

  *Default* → `0.999`

  Forgetting factor applied to the frequency estimates to reduce the impact of old items. The value of `fading_factor` must be in \\((0, 1\]\\).

## Examples

```python
import random
import string
from river import sketch

rng = random.Random(42)
hh = sketch.HeavyHitters()
```

We will feed the counter with printable ASCII characters:

```python
for _ in range(10_000):
    hh.update(rng.choice(string.printable))
```

We can retrieve estimates of the `n` top elements and their frequencies. Let's try `n=3`

```python
hh.most_common(3)
```

```text
[(',', 122.099142...), ('[', 116.049510...), ('W', 115.013402...)]
```

We can also access estimates of individual elements:

```python
hh['A']
```

```text
99.483575...
```

Unobserved elements are handled just fine:

```python
hh[(1, 2, 3)]
```

```text
0.0
```

## Methods

> [!NOTE]
>
> **most\_common**

> [!NOTE]
>
> **update**

[^/dev/api/sketch/HeavyHitters/#1]: Veloso, B., Tabassum, S., Martins, C., Espanha, R., Azevedo, R., &amp; Gama, J. (2020). Interconnect bypass fraud detection: a case study. Annals of Telecommunications, 75(9), 583-596.
