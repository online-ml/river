# TFIDF

Computes TF-IDF values from sentences.

The TF-IDF formula is the same one as scikit-learn. The only difference is the fact that the document frequencies are determined online, whereas in a batch setting they can be determined by performing an initial pass through the data.

Note that the parameters are identical to those of [`feature_extraction.BagOfWords`](<https://riverml.xyz/dev/api/feature-extraction/BagOfWords>).

## Parameters

- **normalize**

  *Default* → `True`

  Whether or not the TF-IDF values by their L2 norm.
- **on**

  *Type* → `str | None`

  *Default* → `None`

  The name of the feature that contains the text to vectorize. If `None`, then the input is treated as a document instead of a set of features.
- **strip\_accents**

  *Default* → `True`

  Whether or not to strip accent characters.
- **lowercase**

  *Default* → `True`

  Whether or not to convert all characters to lowercase.
- **preprocessor**

  *Type* → `typing.Callable | None`

  *Default* → `None`

  An optional preprocessing function which overrides the `strip_accents` and `lowercase` steps, while preserving the tokenizing and n-grams generation steps.
- **stop\_words**

  *Type* → `set[str] | None`

  *Default* → `None`

  An optional set of tokens to remove.
- **tokenizer\_pattern**

  *Default* → `(?u)\b\w[\w\-]+\b`

  The tokenization pattern which is used when no `tokenizer` function is passed. A single capture group may optionally be specified.
- **tokenizer**

  *Type* → `typing.Callable | None`

  *Default* → `None`

  A function used to convert preprocessed text into a `dict` of tokens. By default, a regex formula that works well in most cases is used.
- **ngram\_range**

  *Default* → `(1, 1)`

  The lower and upper boundary of the range n-grams to be extracted. All values of n such that `min_n <= n <= max_n` will be used. For example an `ngram_range` of `(1, 1)` means only unigrams, `(1, 2)` means unigrams and bigrams, and `(2, 2)` means only bigrams. Only works if `tokenizer` is not set to `False`.

## Attributes

- **dfs** (`collections.defaultdict)`)

  Document counts.
- **n** (`int`)

  Number of scanned documents.

## Examples

```python
from river import feature_extraction

tfidf = feature_extraction.TFIDF()

corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]

for sentence in corpus:
    tfidf.learn_one(sentence)
    print(tfidf.transform_one(sentence))
```

```text
{'this': 0.447, 'is': 0.447, 'the': 0.447, 'first': 0.447, 'document': 0.447}
{'this': 0.333, 'document': 0.667, 'is': 0.333, 'the': 0.333, 'second': 0.469}
{'and': 0.497, 'this': 0.293, 'is': 0.293, 'the': 0.293, 'third': 0.497, 'one': 0.497}
{'is': 0.384, 'this': 0.384, 'the': 0.384, 'first': 0.580, 'document': 0.469}
```

In the above example, a string is passed to `transform_one`. You can also indicate which field to access if the string is stored in a dictionary:

```python
tfidf = feature_extraction.TFIDF(on='sentence')

for sentence in corpus:
    x = {'sentence': sentence}
    tfidf.learn_one(x)
    print(tfidf.transform_one(x))
```

```text
{'this': 0.447, 'is': 0.447, 'the': 0.447, 'first': 0.447, 'document': 0.447}
{'this': 0.333, 'document': 0.667, 'is': 0.333, 'the': 0.333, 'second': 0.469}
{'and': 0.497, 'this': 0.293, 'is': 0.293, 'the': 0.293, 'third': 0.497, 'one': 0.497}
{'is': 0.384, 'this': 0.384, 'the': 0.384, 'first': 0.580, 'document': 0.469}
```

In a mini-batch setting, `learn_many` updates the document frequencies from a column of text, and `transform_many` builds a TF-IDF dataframe. The input may be a series of documents, or a dataframe together with the `on` parameter, and any narwhals-supported backend works (pandas, polars, pyarrow, ...). The output matches the input's backend; for pandas it is a memory- efficient sparse dataframe, shown densified here for readability:

```python
import pandas as pd
X = pd.Series(
    ['This is the first document', 'This document is the second document'],
    index=['doc1', 'doc2'],
)

tfidf = feature_extraction.TFIDF()
tfidf.learn_many(X)
tfidf.transform_many(X).sparse.to_dense().round(3)
```

```text
       this     is    the  first  document  second
doc1  0.409  0.409  0.409  0.575     0.409   0.000
doc2  0.334  0.334  0.334  0.000     0.668   0.469
```

## Methods

> [!NOTE]
>
> **learn\_many**

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
> **process\_many**

> [!NOTE]
>
> **process\_text**

> [!NOTE]
>
> **transform\_many**
>
> Transform a column of text into a TF-IDF dataframe.
>
> Accepts any narwhals-supported eager backend (pandas, polars, pyarrow, ...), either as a series of strings or as a dataframe whose `on` column holds the text. The output matches the input backend: a sparse dataframe for pandas, a dense one otherwise.
>
> **Parameters**
>
> - **X** — `IntoSeries | IntoDataFrame`

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
