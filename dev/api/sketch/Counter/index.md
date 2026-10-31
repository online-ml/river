# Counter

Counting using the Count-Min Sketch (CMS) algorithm.

Contrary to an exhaustive approach, e.g., using a `collections.Counter`, CMS uses a limited and fixed amount of memory. The CMS algorithm uses a sketch structure consisting of a matrix \\(w \\times d\\).

These dimensions are obtained via:

- \\(w = \\lceil \\frac{e}{\\epsilon} \\rceil\\), where \\(e\\) is the Euler number.
- \\(d = \\lceil \\ln\\left(\\frac{1}{\\delta} \\right) \\rceil\\).

Decreasing the values of \\(\\epsilon\\) (`epsilon`) and \\(\\delta\\) (`delta`) increase the accuracy of the algorithm, at the cost of increased memory usage. The values of `w` and `d` control the hash tables' capability and the amount of hash collisions, respectively.

CMS works by keeping `d` hash tables with `w` slots each. Elements are mapped to a slot in each hash table. These tables store the counting estimates. This implementation assumes the turnstile case described in the paper, i.e., count values and updates can be negative.

The count values obtained by CMS are always overestimates. Suppose \\(c\_i\\) and \\(\\hat{c}\_i\\) are the ground truth and estimated count values, respectively, for a given element \\(i\\). CMS guarantees that \\(c\_i \\le \\hat{c}\_i\\) and, with probability \\(1 - \\delta\\), \\(\\hat{c}\_i \\le c\_i + \\epsilon||\\mathbf{c}||\_1\\). In the expression, \\(||\\mathbf{c}||\_1 = \\sum\_i |c\_i|\\).

## Parameters

- **epsilon**

  *Type* → `float`

  *Default* → `0.1`

  The approximation error parameter. The error in answering a query is within a factor of `epsilon` with probability `delta`.
- **delta**

  *Type* → `float`

  *Default* → `0.05`

  A query estimates have a probability of `1 - delta` of having errors which are a factor of `epsilon`. See the CMS description above for more details.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random seed for reproducibility.

## Attributes

- **n\_slots**

  The number of slots in each hash table.
- **n\_tables**

  The number of stored hash tables.

## Examples

```python
import collections
from river import sketch

cms = sketch.Counter(epsilon=0.005, seed=0)

rng = random.Random(7)

counter = collections.Counter()
```

We can check the number of slots per hash table:

```python
cms.n_slots
```

```text
544
```

And the number of hash tables:

```python
cms.n_tables
```

```text
3
```

Let's compare the sketch against a brute force approach:

```python
vals = []
for _ in range(10000):
    v = rng.randint(-1000, 1000)
    cms.update(v)
    counter[v] += 1
    vals.append(v)
```

Now, we can compare the estimates of CMS against the exhaustive counting strategy:

```python
counter[7]
```

```text
5
```

```python
cms[7]
```

```text
12
```

```python
counter[532]
```

```text
4
```

```python
cms[532]
```

```text
15
```

Keep in mind that CMS is an approximate sketch algorithm. Counting estimates for unseen values might not be always reliable:

```python
cms[1001]
```

```text
9
```

We can check the number of elements stored by each approach:

```python
len(counter), len(cms)
```

```text
(1982, 1632)
```

And also retrieve the total sum of counts:

```python
cms.total()
```

```text
10000
```

We can decrease the error by allocating more memory in the CMS:

```python
cms_a = sketch.Counter(epsilon=0.001, delta=0.01, seed=0)
for v in vals:
    cms_a.update(v)

cms_a[7]
```

```text
5
```

```python
cms_a[532]
```

```text
4
```

We can also obtain estimates of the dot product between two instances of `river.collections.Counter`. This could be useful, for instance, to estimate the cosine distance between the data monitored in two different counter sketch instances. Suppose we create another CMS instance (the number of slots and hash tables must match) that monitors another sample of the same data generating process:

```python
cms_b = sketch.Counter(epsilon=0.001, delta=0.01, seed=7)

for _ in range(10000):
    v = rng.randint(-1000, 1000)
    cms_b.update(v)
```

Now, we can define a cosine distance function:

```python
def cosine_dist(cms_a, cms_b):
    num = cms_a @ cms_b
    den = math.sqrt(cms_a @ cms_a) * math.sqrt(cms_b @ cms_b)
    return num / den
```

And use it to calculate the cosine distance between the elements monitored in `cms_a` and `cms_b`:

```python
cosine_dist(cms_a, cms_b)
```

```text
0.175363...
```

## Methods

> [!NOTE]
>
> **total**
>
> Return the total count.

> [!NOTE]
>
> **update**

## References

- [Cormode, G., &amp; Muthukrishnan, S. (2005). An improved data stream summary: the count-min sketch and its applications. Journal of Algorithms, 55(1), 58-75.](<https://www.cse.unsw.edu.au/~cs9314/07s1/lectures/Lin_CS9314_References/cm-latin.pdf>)
- [Count-Min Sketch](<https://florian.github.io//count-min-sketch/>)
- [Hash functions family generator in Python](<https://stackoverflow.com/questions/2255604/hash-functions-family-generator-in-python>)
