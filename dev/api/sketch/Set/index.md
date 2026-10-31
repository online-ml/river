# Set

Approximate tracking of observed items using Bloom filters.

Bloom filters enable using a limited amount of memory to check whether a given item was already observed in a stream. They can be used similarly to Python's built-in sets with the difference that items are not explicitly stored. For that reason, element removal and set difference are not currently supported.

Bloom filters store a bit array and map incoming items to `k` index positions in the such array. The selected positions are set to `True`. Therefore, a binary code representation is created for each item. Membership works by projecting the query item and checking if every position of its binary code is `True`. If that is not the case, the item was not observed yet. A nice property of Bloom filters is that they do not yield false negatives: unobserved items might be signalized as observed, but observed items are never signalized as unobserved.

If more than one item has the same binary code, i.e., hash collisions happen, the accuracy of the Bloom filter decreases, and false positives are produced. For instance, a previously unobserved item is signalized as observed. Increasing the size of the binary array and the value of `k` increase the filter's accuracy as hash collisions are avoided. Nonetheless, even using an increased number of hash functions, hash collisions will frequently happen if the array capacity is too small. The length of the bit array and the number of hash functions are inferred automatically from the supplied `capacity` and `fp_rate`.

## Parameters

- **capacity**

  *Type* → `int`

  *Default* → `2048`

  The maximum capacity of the Bloom filter, i.e., the maximum number of distinct items to store given the selected `fp_rate`.
- **fp\_rate**

  *Type* → `float`

  *Default* → `0.01`

  The allowed rate of false positives. The probability of obtaining a true positive is `1 - fp_rate`.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random seed for reproducibility.

## Attributes

- **n\_bits**

  Return the size of the binary array used by the Bloom filter.
- **n\_hash**

  Return the number of used hash functions.

## Examples

```python
import random
from river import sketch

rng = random.Random(42)
s_set = sketch.Set(capacity=100, seed=0)
```

We can retrieve the number of selected hash functions:

```python
s_set.n_hash
```

```text
7
```

And the size of the binary array used by the Bloom filter:

```python
s_set.n_bits
```

```text
959
```

We can add new items and check for membership using the same calls used by Python's standard sets:

```python
for _ in range(1000):
    s_set.add(rng.randint(0, 200))

1 in s_set
```

```text
True
```

False positives might happen if the capacity is not large enough:

```python
-10 in s_set
```

```text
True
```

Iterables can also be supplied to perform multiple updates with a single call to `update`:

```python
s_set = s_set.update([1, 2, 3, 4, 5, 6, 7])
```

We can also combine instances of [`sketch.Set`](<https://riverml.xyz/dev/api/sketch/Set>) using the intersection and union operations, as long as they share the same hash functions and capability. In other words, all they hyperparameters match. Let's create two instances that will monitor different portions of a stream of random numbers:

```python
s1 = sketch.Set(seed=8)
s2 = sketch.Set(seed=8)

for _ in range(1000):
    s1.add(rng.randint(0, 5000))

for _ in range(1000):
    s2.add(rng.randint(0, 5000))

43 in s1
```

```text
True
```

```python
43 in s2
```

```text
False
```

We can get the intersection between the two instances by using:

```python
s_intersection = s1 & s2
43 in s_intersection
```

```text
False
```

We can also obtain the set union:

```python
s_union = s1 | s2

43 in s_union
```

```text
True
```

The same effect of the non-inplace dunder methods can be achieved via explicit method calls:

```python
43 in s1.intersection(s2)
```

```text
False
```

```python
43 in s1.union(s2)
```

```text
True
```

## Methods

> [!NOTE]
>
> **add**

> [!NOTE]
>
> **intersection**
>
> Set intersection.
>
> Return a new instance that results from the set intersection between the current `Set` object and `other`. Dunder operators can be used to replace the method call, i.e., `a &= b` and `a & b` for inplace and non-inplace intersections, respectively.
>
> **Parameters**
>
> - **other** — `Set`

> [!NOTE]
>
> **union**
>
> Set union.
>
> Return a new instance that results from the set union between the current `Set` object and `other`. Dunder operators can be used to replace the method call, i.e., `a |= b` and `a | b` for inplace and non-inplace unions, respectively.
>
> **Parameters**
>
> - **other** — `Set`

> [!NOTE]
>
> **update**

## Notes

This implementation uses an integer to represent the binary array. Bitwise operations are performed in the integer to reflect the Bloom filter updates.

## References

- [Florian Hartmann's blog article on Bloom Filters](<https://florian.github.io/bloom-filters/>).
- [Wikipedia entry on Bloom filters](<https://en.wikipedia.org/wiki/Bloom_filter>).
