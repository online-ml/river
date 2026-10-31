# LDA

Online Latent Dirichlet Allocation with Infinite Vocabulary.

Latent Dirichlet allocation (LDA) is a probabilistic approach for exploring topics in document collections. The key advantage of this variant is that it assumes an infinite vocabulary, meaning that the set of tokens does not have to known in advance, as opposed to the [implementation from sklearn](<https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html>) The results produced by this implementation are identical to those from [the original implementation](<https://github.com/kzhai/PyInfVoc>) proposed by the method's authors.

This class takes as input token counts. Therefore, it requires you to tokenize beforehand. You can do so by using a [`feature_extraction.BagOfWords`](<https://riverml.xyz/dev/api/feature-extraction/BagOfWords>) instance, as shown in the example below.

## Parameters

- **n\_components**

  *Default* → `10`

  Number of topics of the latent Drichlet allocation.
- **number\_of\_documents**

  *Default* → `1000000.0`

  Estimated number of documents.
- **alpha\_theta**

  *Default* → `0.5`

  Hyper-parameter of the Dirichlet distribution of topics.
- **alpha\_beta**

  *Default* → `100.0`

  Hyper-parameter of the Dirichlet process of distribution over words.
- **tau**

  *Default* → `64.0`

  Learning inertia to prevent premature convergence.
- **kappa**

  *Default* → `0.75`

  The learning rate kappa controls how quickly new parameters estimates replace the old ones. kappa ∈ (0.5, 1\] is required for convergence.
- **vocab\_prune\_interval**

  *Default* → `10`

  Interval at which to refresh the words topics distribution.
- **number\_of\_samples**

  *Default* → `10`

  Number of iteration to computes documents topics distribution.
- **ranking\_smooth\_factor**

  *Default* → `1e-12`
- **burn\_in\_sweeps**

  *Default* → `5`

  Number of iteration necessaries while analyzing a document before updating document topics distribution.
- **maximum\_size\_vocabulary**

  *Default* → `4000`

  Maximum size of the stored vocabulary.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random number seed used for reproducibility.

## Attributes

- **counter** (`int`)

  The current number of observed documents.
- **truncation\_size\_prime** (`int`)

  Number of distincts words stored in the vocabulary. Updated before processing a document.
- **truncation\_size** (`int`)

  Number of distincts words stored in the vocabulary. Updated after processing a document.
- **word\_to\_index** (`dict`)

  Words as keys and indexes as values.
- **index\_to\_word** (`dict`)

  Indexes as keys and words as values.
- **nu\_1** (`dict`)

  Weights of the words. Component of the variational inference.
- **nu\_2** (`dict`)

  Weights of the words. Component of the variational inference.

## Examples

```python
from river import compose
from river import feature_extraction
from river import preprocessing

X = [
   'weather cold',
   'weather hot dry',
   'weather cold rainy',
   'weather hot',
   'weather cold humid',
]

lda = compose.Pipeline(
    feature_extraction.BagOfWords(),
    preprocessing.LDA(
        n_components=2,
        number_of_documents=60,
        seed=42
    )
)

for x in X:
    lda.learn_one(x)
    topics = lda.transform_one(x)
    print(topics)
```

```text
{0: 0.5, 1: 2.5}
{0: 2.499..., 1: 1.5}
{0: 0.5, 1: 3.5}
{0: 0.5, 1: 2.5}
{0: 1.5, 1: 2.5}
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
> **learn\_transform\_one**
>
> Equivalent to `lda.learn_one(x).transform_one(x)`s, but faster.
>
> **Parameters**
>
> - **x** — `dict`
>
> **Returns**
>
> `dict`: Component attributions for the input document.

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

- [Zhai, K. and Boyd-Graber, J., 2013, February. Online latent Dirichlet allocation with infinite vocabulary. In International Conference on Machine Learning (pp. 561-569).](<http://proceedings.mlr.press/v28/zhai13.pdf>)
- [PyInfVoc on GitHub](<https://github.com/kzhai/PyInfVoc>)
