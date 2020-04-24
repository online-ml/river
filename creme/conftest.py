collect_ignore = []

try:
    import pandas
except ImportError:
    collect_ignore.append('stream/iter_pandas.py')

try:
    import sklearn
except ImportError:
    collect_ignore.append('compat/sklearn.py')
    collect_ignore.append('compat/test_sklearn.py')

try:
    import surprise
except ImportError:
    collect_ignore.append('reco/surprise.py')

try:
    import torch
except ImportError:
    collect_ignore.append('compat/pytorch.py')

try:
    import vaex
except ImportError:
    collect_ignore.append('stream/iter_vaex.py')
