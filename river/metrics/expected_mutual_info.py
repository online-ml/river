from __future__ import annotations

from river import metrics
from river._river_rust import stats as _rust_stats


def expected_mutual_info(confusion_matrix: metrics.ConfusionMatrix) -> float:
    """Expected Mutual Information between two clusterings.

    Used internally by `metrics.AdjustedMutualInfo`; see its docstring for the
    role this term plays in the AMI formula.
    """
    a = [int(v) for v in confusion_matrix.sum_row.values() if v]
    b = [int(v) for v in confusion_matrix.sum_col.values() if v]
    return _rust_stats.expected_mutual_info(confusion_matrix.n_samples, a, b)
