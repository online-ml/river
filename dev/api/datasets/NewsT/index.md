# NewsT

News-T text stream clustering dataset.

The data contains 11,109 preprocessed news titles assigned to 152 topics. It is derived from GoogleNews and reordered into 16 time points so that topics emerge and disappear over time. The available feature is the preprocessed title, and the target is the topic identifier.

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

- [Dynamic clustering for short text stream based on Dirichlet process](<https://doi.org/10.1186/s40537-021-00464-8>)
- [Textual One-Pass Stream Clustering with Automated Distance Threshold Adaption](<https://doi.org/10.1007/978-3-031-21743-2_1>)
