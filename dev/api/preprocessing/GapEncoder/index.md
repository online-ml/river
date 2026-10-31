# GapEncoder

Online Gamma-Poisson encoder for fuzzy string categories.

This is an online version of [skrub's `GapEncoder`](<https://skrub-data.org/stable/reference/generated/skrub.GapEncoder.html>), learning one string at a time instead of in batches. Each string is turned into its character n-gram counts `v`, and we look for a small set of latent topics that explain those counts: `v ≈ h @ W`, where `W` holds the topic/n-gram weights and `h` says how strongly each topic is activated by the string. The counts are treated as Poisson with a Gamma prior on `h`, and both `h` and `W` are refined with multiplicative updates as data comes in.

The point is fuzzy matching. Strings that mean the same thing usually share most of their n-grams ("london", "London, UK", "Lomdon"), so they light up the same topics even when they have no exact token in common. That makes the encoder handy for messy categorical text: hand-typed city names, job titles, chat messages full of typos, and so on.

The n-gram vocabulary is not fixed ahead of time; it grows as new strings show up, the same way [`preprocessing.LDA`](<https://riverml.xyz/dev/api/preprocessing/LDA>) grows its word vocabulary. Topics are kept up to date through two accumulators `A` and `B`, with a `half_life` (in samples) that lets old observations decay. `transform_one` is read-only: it only looks at n-grams it has already seen and never changes the model.

## Parameters

- **n\_components**

  *Type* → `int`

  *Default* → `10`

  Number of latent topics.
- **on**

  *Type* → `str | None`

  *Default* → `None`

  The name of the feature that contains the text to encode. If `None`, then each `learn_one` and `transform_one` should treat `x` as a `str` and not as a `dict`.
- **strip\_accents**

  *Type* → `bool`

  *Default* → `True`

  Whether or not to strip accent characters.
- **lowercase**

  *Type* → `bool`

  *Default* → `True`

  Whether or not to convert all characters to lowercase.
- **ngram\_range**

  *Type* → `tuple[int, int]`

  *Default* → `(2, 4)`

  The lower and upper boundary of the range of character n-grams to be extracted. All values of n such that `min_n <= n <= max_n` will be used.
- **gamma\_shape\_prior**

  *Type* → `float`

  *Default* → `1.1`

  Shape parameter of the Gamma prior on the activations.
- **gamma\_scale\_prior**

  *Type* → `float`

  *Default* → `1.0`

  Scale parameter of the Gamma prior on the activations.
- **half\_life**

  *Type* → `float`

  *Default* → `1000.0`

  Forgetting horizon for the topics, in number of samples. A sample's influence on the topic accumulators halves every `half_life` observations, i.e. the per-sample decay is `0.5 ** (1 / half_life)`. Larger values keep a longer memory and let the topics build up stable global structure; smaller values adapt faster to drift. Use `float('inf')` to never forget. This replaces the batch implementation's `rho`, which is not meaningful when updating one sample at a time (`rho = 0.5 ** (1 / half_life)`).
- **max\_iter\_e\_step**

  *Type* → `int`

  *Default* → `10`

  Number of multiplicative iterations used to fit the activations of a sample during `learn_one`. `transform_one` always iterates until convergence (up to 100 iterations).
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random number seed used for reproducibility. New vocabulary columns of `W` are initialized with Gamma-distributed random draws.

## Attributes

- **vocab** (`dict`)

  Maps each seen n-gram to its column index.
- **W** (`np.ndarray`)

  Topic/n-gram weights, shape `(n_components, len(vocab))`. Rows sum to 1.
- **A** (`np.ndarray`)

  Numerator accumulator of the topic updates, same shape as `W`.
- **B** (`np.ndarray`)

  Denominator accumulator of the topic updates, shape `(n_components, 1)`.

## Examples

Say people type in city names by hand. The same city shows up spelled several different ways, with typos and extra bits, and no two spellings need to share a whole word. The encoder still groups the variants under the same topic:

```python
from river import preprocessing

enc = preprocessing.GapEncoder(n_components=2, seed=42)

X = ["london", "London", "London, UK", "Lomdon", "paris", "Paris", "Paris, France", "pqris"]
for _ in range(10):
    for x in X:
        enc.learn_one(x)
```

Each variant of a city activates the topic of that city, even with typos never seen during training:

```python
for x in ["London, UK", "Lndon", "Paris, France", "Pariss"]:
    topics = enc.transform_one(x)
    print(f"{x} -> topic {max(topics, key=topics.get)}")
```

```text
London, UK -> topic 1
Lndon -> topic 1
Paris, France -> topic 0
Pariss -> topic 0
```

Because it outputs a numeric dict, a `GapEncoder` drops straight into a classification pipeline. Here it learns to tell UK cities from French ones from their messy spellings, scored prequentially (test-then-train):

```python
from river import evaluate, linear_model, metrics

cities = [
    ("london", True), ("Londn", True), ("LONDON", True), ("londonn", True),
    ("manchester", True), ("manchestr", True), ("leeds", True), ("leedss", True),
    ("paris", False), ("Pariss", False), ("PARIS", False), ("pqris", False),
    ("lyon", False), ("lyonn", False), ("nice", False), ("niice", False),
]

model = preprocessing.GapEncoder(n_components=5, seed=42) | linear_model.LogisticRegression()
evaluate.progressive_val_score(cities * 15, model, metrics.Accuracy())
```

```text
Accuracy: 83.75%
```

## Methods

> [!NOTE]
>
> **learn\_one**
>
> Update with a set of features `x`.
>
> A lot of transformers don't actually have to do anything during the `learn_one` step because they are stateless. For this reason the default behavior of this function is to do nothing. Transformers that however do something during the `learn_one` can override this method.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, Any]`

> [!NOTE]
>
> **transform\_one**
>
> Transform a set of features `x`.
>
> **Parameters**
>
> - **x** — `dict[base.typing.FeatureName, Any]`
>
> **Returns**
>
> `dict&#91;base.typing.FeatureName, Any&#93;`: The transformed values.

## References

- [Cerda, P. and Varoquaux, G., 2020. Encoding high-cardinality string categorical variables. IEEE Transactions on Knowledge and Data Engineering.](<https://inria.hal.science/hal-02171256v4>)
- [skrub's GapEncoder](<https://skrub-data.org/stable/reference/generated/skrub.GapEncoder.html>)
