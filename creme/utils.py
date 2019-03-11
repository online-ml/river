def normalize_y_pred(y_pred):
    """Normalizes a dictionary of predicted probabilities, in-place."""
    total = sum(y_pred.values())

    if total > 0:
        for i in y_pred:
            y_pred[i] /= total
    else:
        for i in y_pred:
            y_pred[i] = 1 / len(y_pred)

    return y_pred
