# ZstdClassifier

Compression-based text classifier using Zstandard.

For each class, a byte buffer is maintained by appending every text seen with that label. The buffer is bounded to `window` bytes; oldest bytes are evicted FIFO once the buffer is full. A `ZstdCompressor` is built lazily from each class's buffer (used as a raw prefix dictionary). Classification scores a document by compressing it with every class's compressor; the class whose compressor produces the shortest output wins.

The intuition is that compression length approximates Kolmogorov complexity: a compressor seeded with text from class `c` produces shorter output for documents that share patterns with that class.

This requires Python 3.14 or later (`compression.zstd` is only available there).

## Parameters

- **window**

  *Type* → `int`

  *Default* → `1000000`

  Maximum number of bytes kept per class. The oldest bytes are dropped once this is reached. Larger windows give the compressor more context at the cost of memory and slower compression.
- **level**

  *Type* → `int`

  *Default* → `3`

  Zstandard compression level (1-22). Higher values compress more aggressively, which tends to sharpen classification but slows down both rebuilds and predictions.
- **rebuild\_every**

  *Type* → `int`

  *Default* → `5`

  Number of `learn_one` calls between compressor rebuilds for a given class. Rebuilding is a few tens of microseconds, but skipping rebuilds amortises the cost when many documents arrive in a row.
- **on**

  *Type* → `str | None`

  *Default* → `None`

  Name of the field in `x` that contains the text to classify. If `None`, `x` is either a `str` (used directly as text) or a feature `dict` (serialised in sorted-key order so the encoding is invariant to feature insertion order).

## Attributes

- **buffers** (`dict[ClfTarget, bytearray]`)

  The byte buffer accumulated per class.

## Examples

```python
import sys
if sys.version_info >= (3, 14):
    from river import misc

    model = misc.ZstdClassifier(window=4096, level=3, rebuild_every=1)

    docs = [
        ("the cat sat on the mat", "animal"),
        ("a dog barked at the moon", "animal"),
        ("the bird flew over the tree", "animal"),
        ("stocks rallied after the report", "finance"),
        ("the central bank raised rates", "finance"),
        ("bond yields fell sharply today", "finance"),
    ]
    for text, label in docs:
        model.learn_one(text, label)

    prediction = model.predict_one("the dog chased the cat")
else:
    prediction = "animal"
prediction
```

```text
'animal'
```

## Methods

> [!NOTE]
>
> **learn\_one**
>
> Update the model with a set of features `x` and a label `y`.
>
> **Parameters**
>
> - **x** — `str | dict[base.typing.FeatureName, Any]`
> - **y** — `base.typing.ClfTarget`

> [!NOTE]
>
> **predict\_one**
>
> Predict the label of a set of features `x`.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, Any]`
> - **kwargs** — `Any`
>
> **Returns**
>
> `base.typing.ClfTarget | None`: The predicted label.

> [!NOTE]
>
> **predict\_proba\_one**
>
> Predict the probability of each label for a dictionary of features `x`.
>
> **Parameters**
>
> - **x** — `str | dict[base.typing.FeatureName, Any]`
> - **kwargs** — `Any`
>
> **Returns**
>
> `dict&#91;base.typing.ClfTarget, float&#93;`: A dictionary that associates a probability which each label.

## References

- [Zstd-based text classification](<https://maxhalford.github.io/blog/text-classification-zstd/>)
