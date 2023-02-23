import numpy as np

from sklearn.metrics import roc_auc_score as sk_rocauc


def roc_auc_score(y_true, y_score):
    """
        This functions is a wrapper to the scikit-learn roc_auc_score function,
        which is used on the test_metrics.py. It was created because the scikit
        version utilizes array of scores and may raise a ValueError if there
        is only one class present in y_true. This wrapper returns 0 if y_true
        has only one class and deals with the scores.
    """
    nonzero = np.count_nonzero(y_true)
    if nonzero == 0 or nonzero == len(y_true):
        return 0

    scores = [s[True] for s in y_score]

    return sk_rocauc(y_true, scores)
