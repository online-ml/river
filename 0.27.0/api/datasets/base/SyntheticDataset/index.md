# SyntheticDataset

A synthetic dataset.

## Parameters

- **task**

  Type of task the dataset is meant for. Should be one of: - "Regression" - "Binary classification" - "Multi-class classification" - "Multi-output binary classification" - "Multi-output regression"
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
