# FileDataset

Base class for datasets that are stored in a local file.

Small datasets that are part of the river package inherit from this class.

## Parameters

- **filename**

  The file's name.
- **directory**

  *Default* → `None`

  The directory where the file is contained. Defaults to the location of the `datasets` module.

## Attributes

- **desc**

  Return the description from the docstring.
- **path**

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
