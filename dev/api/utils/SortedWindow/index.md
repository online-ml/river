# SortedWindow

Sorted running window data structure.

## Parameters

- **size**

  *Type* → `int`

  Size of the window to compute the rolling quantile.

## Attributes

- **size**

## Examples

```python
from river import utils

window = utils.SortedWindow(size=3)

for i in reversed(range(9)):
    window.append(i)
    print(window)
```

```text
[8]
[7, 8]
[6, 7, 8]
[5, 6, 7]
[4, 5, 6]
[3, 4, 5]
[2, 3, 4]
[1, 2, 3]
[0, 1, 2]
```

## Methods

## References

- [Left sorted inserts in Python](<https://stackoverflow.com/questions/8024571/insert-an-item-into-sorted-list-in-python>)
