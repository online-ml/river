# CriteoAds

Criteo Display Advertising Challenge.

This is a 100,000-row sample of the Criteo Display Advertising Challenge dataset: real ad-impression logs where the goal is to predict whether an ad was clicked. Each row has a binary `click` target, 13 integer features (`I1` to `I13`, mostly anonymised counts, frequently missing) and 26 categorical features (`C1` to `C26`, hashed to opaque strings). The categorical fields are extremely high-cardinality, which makes this a natural fit for one-hot models such as [`linear_model.AdPredictor`](<https://riverml.xyz/dev/api/linear-model/AdPredictor>).

Integer features are returned as `int` (or `None` when missing) and categorical features as `str` (or `None` when missing).

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

- [Display Advertising Challenge - Criteo AI Lab](<https://ailab.criteo.com/display-advertising-challenge-criteo/>)
- [Kaggle competition page](<https://www.kaggle.com/c/criteo-display-ad-challenge>)
