# VectorDict

A dictionary-like object that supports vector-like operations.

Supports addition (+), subtraction (-), multiplication (\*) and division (/) with a VectorDict or a scalar. Supports dot product (@) with a VectorDict. A scalar is any object that supports the four arithmetic operations with the dictionary's values.

If mask is not None, any key which is not contained in mask is said to be masked while other keys are said to be unmasked. If mask is None, any key is said to be unmasked.

If default\_factory is not None, it is called whenever an unmasked missing key is accessed, either externally with **getitem** or internally as part of an element-wise numeric operation such as addition, and the result is inserted as the value for that key. If a masked key, or an unmasked missing key when default\_factory is None, is accessed externally through **getitem**, a KeyError exception is raised, and if it is accessed internally as part of an operation, its value is taken as 0, but is not inserted for that key.

If copy is True, a copy of data and mask will be made if not None and these arguments will not be modified. If copy is False, references to data and mask will be used if not None. This means that the argument data may be modified, although only on unmasked keys, and that external modifications of data and mask will affect the internal operations.

## Parameters

- **data**

  *Default* → `None`

  A VectorDict or dict to initialize key-values from, or None.
- **default\_factory**

  *Default* → `None`

  A callable returning a scalar, or None.
- **mask**

  *Default* → `None`

  A VectorDict or set-like object such that keys not in mask will not be considered in operations and will always result in a KeyError if accessed by **getitem**, or None.
- **copy**

  *Default* → `False`

  If data and/or mask are specified, whether to store a copy of the underlying dictionaries or references at initialization.

## Methods

> [!NOTE]
>
> **abs**

> [!NOTE]
>
> **clear**

> [!NOTE]
>
> **dot**

> [!NOTE]
>
> **euclidean\_distance**

> [!NOTE]
>
> **from\_scaled**

> [!NOTE]
>
> **get**

> [!NOTE]
>
> **iadd\_scaled**

> [!NOTE]
>
> **isub\_scaled**

> [!NOTE]
>
> **items**

> [!NOTE]
>
> **keys**

> [!NOTE]
>
> **max**

> [!NOTE]
>
> **maximum**

> [!NOTE]
>
> **min**

> [!NOTE]
>
> **minimum**

> [!NOTE]
>
> **pop**

> [!NOTE]
>
> **popitem**

> [!NOTE]
>
> **setdefault**

> [!NOTE]
>
> **to\_dict**

> [!NOTE]
>
> **to\_numpy**

> [!NOTE]
>
> **update**

> [!NOTE]
>
> **update\_ema**

> [!NOTE]
>
> **values**

> [!NOTE]
>
> **with\_mask**
