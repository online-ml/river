# Link

A link joins two univariate statistics as a sequence.

This can be used to pipe the output of one statistic to the input of another. This can be used, for instance, to calculate the mean of the variance of a variable. It can also be used to compute shifted statistics by piping statistics with an instance of [`stats.Shift`](<https://riverml.xyz/dev/api/stats/Shift>).

Note that a link is not meant to be instantiated via this class definition. Instead, users can link statistics together via the `|` operator.

## Parameters

- **left**

  *Type* → `Univariate[_OutT]`
- **right**

  *Type* → `Univariate[_ChainedOutT]`

  The output from `left`'s `get` method is passed to `right`'s `update` method if `left`'s `get` method doesn't produce `None.`

## Attributes

- **name**

## Examples

```python
from river import stats
stat = stats.Shift(1) | stats.Mean()
```

No values have been seen, therefore `get` defaults to the initial value of [`stats.Mean`](<https://riverml.xyz/dev/api/stats/Mean>), which is 0.

```python
stat.get()
```

```text
0.0
```

Let us now call `update`.

```python
stat.update(1)
```

The output from `get` will still be 0. The reason is that [`stats.Shift`](<https://riverml.xyz/dev/api/stats/Shift>) has not enough values, and therefore outputs its default value, which is `None`. The [`stats.Mean`](<https://riverml.xyz/dev/api/stats/Mean>) instance is therefore not updated.

```python
stat.get()
```

```text
0.0
```

On the next call to `update`, the [`stats.Shift`](<https://riverml.xyz/dev/api/stats/Shift>) instance has seen enough values, and therefore the mean can be updated. The mean is therefore equal to 1, because that's the only value from the past.

```python
stat.update(3)
stat.get()
```

```text
1.0
```

On the subsequent call to update, the mean will be updated with the value 3.

```python
stat.update(4)
stat.get()
```

```text
2.0
```

Note that composing statistics returns a new statistic with its own name.

```python
stat.name
```

```text
'mean_of_shift_1'
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
> - **x** — `Any`
