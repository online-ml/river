# shuffle

Shuffles a stream of data.

This works by maintaining a buffer of elements. The first `buffer_size` elements are stored in memory. Once the buffer is full, a random element inside the buffer is yielded. Every time an element is yielded, the next element in the stream replaces it and the buffer is sampled again. Increasing `buffer_size` will improve the quality of the shuffling.

If you really want to stream over your dataset in a "good" random order, the best way is to split your dataset into smaller datasets and loop over them in a round-robin fashion. You may do this by using the `roundrobin` recipe from the `itertools` module.

## Parameters

- **stream**

  *Type* → `typing.Iterable[T]`

  The stream to shuffle.
- **buffer\_size**

  *Type* → `int`

  The size of the buffer which contains the elements help in memory. Increasing this will increase randomness but will incur more memory usage.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random seed used for sampling.

## Examples

```python
from river import stream

for i in stream.shuffle(range(15), buffer_size=5, seed=42):
    print(i)
```

```text
0
5
2
1
8
9
6
4
11
12
10
7
14
13
3
```

## References

- [Visualizing TensorFlow's streaming shufflers](<http://www.moderndescartes.com/essays/shuffle_viz/>)
