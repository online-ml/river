# BagOfWords

Counts tokens in sentences.

This transformer can be used to counts tokens in a given piece of text. It takes care of normalizing the text before tokenizing it. In mini-batch settings, this transformers allows to convert a series of pandas of text into sparse dataframe.

Note that the parameters are identical to those of [`feature_extraction.TFIDF`](<https://riverml.xyz/dev/api/feature-extraction/TFIDF>).

## Parameters

- **on**

  *Type* → `str | None`

  *Default* → `None`

  The name of the feature that contains the text to vectorize. If `None`, then each `learn_one` and `transform_one` will assume that each `x` that is provided is a `str`, andnot a `dict`.
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

  The lower and upper boundary of the range n-grams to be extracted. All values of n such that `min_n <= n <= max_n` will be used. For example an `ngram_range` of `(1, 1)` means only unigrams, `(1, 2)` means unigrams and bigrams, and `(2, 2)` means only bigrams.

## Examples

By default, `BagOfWords` will take as input a sentence, preprocess it, tokenize the preprocessed text, and then return a `collections.Counter` containing the number of occurrences of each token.

```python
from river import feature_extraction as fx

corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]

bow = fx.BagOfWords()

for sentence in corpus:
    print(bow.transform_one(sentence))
```

```text
{'this': 1, 'is': 1, 'the': 1, 'first': 1, 'document': 1}
{'this': 1, 'document': 2, 'is': 1, 'the': 1, 'second': 1}
{'and': 1, 'this': 1, 'is': 1, 'the': 1, 'third': 1, 'one': 1}
{'is': 1, 'this': 1, 'the': 1, 'first': 1, 'document': 1}
```

Note that `learn_one` does not have to be called because `BagOfWords` is stateless. You can call it but it won't do anything.

In the above example, a string is passed to `transform_one`. You can also indicate which field to access if the string is stored in a dictionary:

```python
bow = fx.BagOfWords(on='sentence')

for sentence in corpus:
    x = {'sentence': sentence}
    print(bow.transform_one(x))
```

```text
{'this': 1, 'is': 1, 'the': 1, 'first': 1, 'document': 1}
{'this': 1, 'document': 2, 'is': 1, 'the': 1, 'second': 1}
{'and': 1, 'this': 1, 'is': 1, 'the': 1, 'third': 1, 'one': 1}
{'is': 1, 'this': 1, 'the': 1, 'first': 1, 'document': 1}
```

The `ngram_range` parameter can be used to extract n-grams (including unigrams):

```python
ngrammer = fx.BagOfWords(ngram_range=(1, 2))

ngrams = ngrammer.transform_one('I love the smell of napalm in the morning')
for ngram, count in ngrams.items():
    print(ngram, count)
```

```text
love 1
the 2
smell 1
of 1
napalm 1
in 1
morning 1
('love', 'the') 1
('the', 'smell') 1
('smell', 'of') 1
('of', 'napalm') 1
('napalm', 'in') 1
('in', 'the') 1
('the', 'morning') 1
```

`BagOfWord` allows to build a term-frequency pandas sparse dataframe with the `transform_many` method.

```python
import pandas as pd
X = pd.Series(['Hello world', 'Hello River'], index = ['river', 'rocks'])
bow = fx.BagOfWords()
bow.transform_many(X=X)
```

```text
       hello  world  river
river      1      1      0
rocks      1      0      1
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
> Transform a column of text into a term-frequency dataframe.
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
