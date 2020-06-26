"""Missing data imputation.

As a convention, missing values are indicated with an explicit `None` value. In other words, they
are present in the dictionary of features. For instance, in the following example, the feature
`shop` has a missing value:

```python
x = {'country': 'Sweden', 'shop': None}
```

"""
from .previous import PreviousImputer
from .stat import StatImputer


__all__ = [
    'PreviousImputer',
    'StatImputer'
]
