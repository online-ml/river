from __future__ import annotations

from river import preprocessing


def test_previous_imputer_does_not_mutate_input():
    imputer = preprocessing.PreviousImputer()
    imputer.learn_one({"x": 5})

    x = {"x": None, "y": 1}
    original = dict(x)
    transformed = imputer.transform_one(x)

    assert x == original, "transform_one mutated its input"
    assert transformed is not x
    assert transformed == {"x": 5, "y": 1}
