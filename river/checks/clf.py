from __future__ import annotations

import copy
import math


def check_predict_proba_one(classifier, dataset):
    """predict_proba_one should return a valid probability distribution and be pure."""

    from river.active.base import ActiveLearningClassifier

    if not hasattr(classifier, "predict_proba_one"):
        return

    for x, y in dataset:
        xx, yy = copy.deepcopy(x), copy.deepcopy(y)

        classifier.learn_one(x, y)
        y_pred = classifier.predict_proba_one(x)

        if isinstance(classifier, ActiveLearningClassifier):
            y_pred, _ = y_pred

        # Check the probabilities are coherent
        assert isinstance(y_pred, dict)
        for proba in y_pred.values():
            assert 0.0 <= proba <= 1.0
        assert math.isclose(sum(y_pred.values()), 1.0)

        # Check predict_proba_one is pure (i.e. x and y haven't changed)
        assert x == xx
        assert y == yy


def check_predict_proba_one_binary(classifier, dataset):
    """predict_proba_one should return a dict with True and False keys."""

    for x, y in dataset:
        y_pred = classifier.predict_proba_one(x)
        classifier.learn_one(x, y)
        assert set(y_pred.keys()) == {False, True}


def check_multiclass_is_bool(model):
    assert isinstance(model._multiclass, bool)


def check_classifier_tracks_seen_labels(classifier, dataset):
    """Every label seen during training should appear in `predict_proba_one`.

    Catches classifiers that silently drop labels they have already observed.
    """

    from river.active.base import ActiveLearningClassifier

    if not hasattr(classifier, "predict_proba_one"):
        return

    seen: set = set()
    last_x = None
    for x, y in dataset:
        classifier.learn_one(x, y)
        seen.add(y)
        last_x = x

    if last_x is None or len(seen) < 2:
        return

    try:
        proba = classifier.predict_proba_one(last_x)
    except NotImplementedError:
        return

    if isinstance(classifier, ActiveLearningClassifier):
        proba, _ = proba

    missing = seen - set(proba.keys())
    assert not missing, (
        f"predict_proba_one is missing labels seen during training: {sorted(missing, key=repr)}"
    )
