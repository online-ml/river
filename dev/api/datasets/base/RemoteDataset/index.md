# RemoteDataset

Base class for datasets that are stored in a remote file.

Medium and large datasets that are not part of the river package inherit from this class.

The filename doesn't have to be provided if unpack is False. Indeed in the latter case the filename will be inferred from the URL.

## Parameters

- **url**

  The URL the dataset is located at.
- **size**

  The expected download size.
- **unpack**

  *Default* → `True`

  Whether to unpack the download or not.
- **filename**

  *Default* → `None`

  An optional name to given to the file if the file is unpacked.

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
