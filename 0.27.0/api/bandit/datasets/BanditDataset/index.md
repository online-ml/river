# BanditDataset

Base class for bandit datasets.

## Parameters

- **n\_features**

  Number of features in the dataset.
- **n\_samples**

  *Default* → `None`

  Number of samples in the dataset.
- **n\_classes**

  *Default* → `None`

  Number of classes in the dataset, only applies to classification datasets.
- **n\_outputs**

  *Default* → `None`

  Number of outputs the target is made of, only applies to multi-output datasets.
- **sparse**

  *Default* → `False`

  Whether the dataset is sparse or not.

## Attributes

- **arms**

  The list of arms that can be pulled.
- **desc**

  Return the description from the docstring.

## Methods

> [!NOTE]
>
> **take**
>
> Iterate over the k samples.
>
> **Parameters**
>
> - **k** — `int`
