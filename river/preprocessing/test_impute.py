from __future__ import annotations

from river import preprocessing


def test_previous_imputer_does_not_mutate_input():
    """transform_one must be pure: it should not modify the caller's dict.

    The purity check in river.checks uses a dataset without missing values, so
    it never exercises the branch that fills a None, which is exactly where the
    mutation happened.
    """
    imputer = preprocessing.PreviousImputer()
    imputer.learn_one({"x": 5})

    x = {"x": None, "y": 1}
    original = dict(x)
    transformed = imputer.transform_one(x)

    assert x == original, "transform_one mutated its input"
    assert transformed is not x
    assert transformed == {"x": 5, "y": 1}
