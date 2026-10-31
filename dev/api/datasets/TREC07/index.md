# TREC07

TREC's 2007 Spam Track dataset.

The data contains 75,419 chronologically ordered items, i.e. 3 months of emails delivered to a particular server in 2007. Spam messages represent 66.6% of the dataset. The goal is to predict whether an email is a spam or not.

The available raw features are: sender, recipients, date, subject, body.

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

- [TREC 2007 Spam Track Overview](<https://trec.nist.gov/pubs/trec16/papers/SPAM.OVERVIEW16.pdf>)
- [Code ran to parse the dataset](<https://gist.github.com/gbolmier/b6a942699aaaedec54041a32e4f34d40>)
