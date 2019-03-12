import math


def softmax(y_pred):
    """Normalizes a dictionary of predicted probabilities, in-place."""
    exp = {c: math.exp(p) for c, p in y_pred.items()}
    total = sum(exp.values())
    return {c: exp[c] / total for c in y_pred}
