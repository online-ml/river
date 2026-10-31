# Insects

Insects dataset.

This dataset has different variants for concept drift evaluation:

- abrupt\_balanced
- abrupt\_imbalanced
- gradual\_balanced
- gradual\_imbalanced
- incremental\_abrupt\_balanced
- incremental\_reoccurring\_balanced
- incremental\_balanced

The number of samples and the difficulty change from one variant to another. The number of classes is always the same (6), except for the last variant (24).

## Parameters

- **variant**

  *Default* → `abrupt_balanced`

  Indicates which variant of the dataset to load. Defaults to "abrupt\_balanced".

## Attributes

- **desc**

  Return the description from the docstring.
- **is\_downloaded**

  Indicate whether or the data has been correctly downloaded.
- **path**

## Methods

> [!NOTE]
>
> **download**

> [!NOTE]
>
> **take**
>
> Iterate over the k samples.
>
> **Parameters**
>
> - **k** — `int`

## References

- [USP DS repository](<https://sites.google.com/view/uspdsrepository>)
- [Souza, V., Reis, D.M.D., Maletzke, A.G. and Batista, G.E., 2020. Challenges in Benchmarking Stream Learning Algorithms with Real-world Data. arXiv preprint arXiv:2005.00113.](<https://arxiv.org/abs/2005.00113>)
