# NUnique

Approximate number of unique values counter.

This is basically an implementation of the HyperLogLog algorithm. Adapted from [`hypy`](<https://github.com/clarkduvall/hypy>). The code is a bit too terse but it will do for now.

## Parameters

- **error\_rate**

  *Type* → `float`

  *Default* → `0.01`

  Desired error rate. Memory usage is inversely proportional to this value.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Set the seed to produce identical results.

## Attributes

- **n\_bits** (`int`)
- **n\_buckets** (`int`)
- **buckets** (`list`)

## Examples

```python
import string
from river import sketch

alphabet = string.ascii_lowercase
n_unique = sketch.NUnique(error_rate=0.2, seed=42)

n_unique.update('a')
n_unique.get()
```

```text
1
```

```python
n_unique.update('b')
n_unique.get()
```

```text
2
```

```python
for letter in alphabet:
    n_unique.update(letter)
n_unique.get()
```

```text
31
```

Lowering the `error_rate` parameter will increase the precision.

```python
n_unique = sketch.NUnique(error_rate=0.01, seed=42)
for letter in alphabet:
    n_unique.update(letter)
n_unique.get()
```

```text
26
```

## Methods

> [!NOTE]
>
> **get**

> [!NOTE]
>
> **update**

## References

- [My favorite algorithm (and data structure): HyperLogLog](<https://odino.org/my-favorite-data-structure-hyperloglog/>)
- [Flajolet, P., Fusy, É., Gandouet, O. and Meunier, F., 2007, June. Hyperloglog: the analysis of a near-optimal cardinality estimation algorithm.](<http://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf>)
