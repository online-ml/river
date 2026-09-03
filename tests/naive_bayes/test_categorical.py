from __future__ import annotations

import typing

import numpy as np
import pandas as pd
import pytest
from sklearn.naive_bayes import CategoricalNB as SklearnCategoricalNB

from river import naive_bayes

if typing.TYPE_CHECKING:
    pass


def _categorical_data():
    X = pd.DataFrame(
        {
            "color": ["red", "yellow", "red", "green"],
            "shape": ["round", "long", "round", "long"],
        }
    )
    y = pd.Series(["apple", "banana", "apple", "banana"])
    return X, y


def _encode_categories(X):
    X_encoded = X.copy()

    for column in X.columns:
        categories = {value: i for i, value in enumerate(X[column].unique())}
        X_encoded[column] = X[column].map(categories)

    return X_encoded


def test_categorical_learn_many_vs_learn_one():
    X, y = _categorical_data()

    model = naive_bayes.CategoricalNB(alpha=1.0)
    batch_model = naive_bayes.CategoricalNB(alpha=1.0)

    for _, row in X.iterrows():
        model.learn_one(row.to_dict(), y.loc[row.name])

    batch_model.learn_many(X, y)

    assert model.class_counts == batch_model.class_counts
    assert model.feature_counts == batch_model.feature_counts
    assert model.feature_values == batch_model.feature_values
    assert model.classes_ == batch_model.classes_

    X_test = pd.DataFrame(
        {
            "color": ["red", "yellow", "green"],
            "shape": ["round", "long", "long"],
        },
        index=["a", "b", "c"],
    )

    batch_scores = batch_model.joint_log_likelihood_many(X_test)

    assert batch_scores.index.tolist() == X_test.index.tolist()
    assert batch_scores.columns.tolist() == model.classes_

    for i, row in X_test.iterrows():
        one = model.joint_log_likelihood(row.to_dict())

        for c in model.classes_:
            assert batch_scores.loc[i, c] == pytest.approx(one[c])


def test_categorical_not_fit():
    model = naive_bayes.CategoricalNB()

    X = pd.DataFrame(
        {
            "color": ["red", "blue"],
            "shape": ["round", "square"],
        },
        index=["river", "rocks"],
    )

    assert model.joint_log_likelihood_many(X).empty


def test_categorical_smoothing():
    model = naive_bayes.CategoricalNB(alpha=1.0)

    model.learn_one({"color": "red"}, "apple")
    model.learn_one({"color": "red"}, "apple")
    model.learn_one({"color": "yellow"}, "banana")

    p = model.p_feature_given_class("color", "blue", "apple")

    assert p > 0
    assert p < 1


def test_categorical_class_probabilities():
    model = naive_bayes.CategoricalNB()

    model.learn_one({"x": "a"}, "yes")
    model.learn_one({"x": "b"}, "yes")
    model.learn_one({"x": "c"}, "no")

    assert model.p_class("yes") == pytest.approx(2 / 3)
    assert model.p_class("no") == pytest.approx(1 / 3)


def test_categorical_predict_proba_many_matches_learn_one():
    X, y = _categorical_data()

    model = naive_bayes.CategoricalNB(alpha=1.0)
    model.learn_many(X, y)

    X_test = pd.DataFrame(
        {
            "color": ["red", "yellow", "green"],
            "shape": ["round", "long", "long"],
        },
        index=["a", "b", "c"],
    )

    batch_proba = model.predict_proba_many(X_test)

    assert batch_proba.index.tolist() == X_test.index.tolist()
    assert batch_proba.columns.tolist() == model.classes_

    for i, row in X_test.iterrows():
        one = model.predict_proba_one(row.to_dict())

        for c in model.classes_:
            assert batch_proba.loc[i, c] == pytest.approx(one[c])


def test_categorical_matches_sklearn():
    X, y = _categorical_data()

    river_model = naive_bayes.CategoricalNB(alpha=1.0)
    river_model.learn_many(X, y)

    X_encoded = _encode_categories(X)

    sklearn_model = SklearnCategoricalNB(alpha=1.0)
    sklearn_model.fit(X_encoded, y)

    river_proba = river_model.predict_proba_many(X)

    sklearn_proba = pd.DataFrame(
        sklearn_model.predict_proba(X_encoded),
        columns=sklearn_model.classes_,
        index=X.index,
    )

    assert river_proba.columns.tolist() == sklearn_proba.columns.tolist()

    assert np.allclose(
        river_proba.to_numpy(),
        sklearn_proba.to_numpy(),
        atol=1e-10,
    )


def test_categorical_joint_log_likelihood_matches_sklearn():
    X, y = _categorical_data()

    river_model = naive_bayes.CategoricalNB(alpha=1.0)
    river_model.learn_many(X, y)

    X_encoded = _encode_categories(X)

    sklearn_model = SklearnCategoricalNB(alpha=1.0)
    sklearn_model.fit(X_encoded, y)

    river_jll = river_model.joint_log_likelihood_many(X)

    sklearn_jll = pd.DataFrame(
        sklearn_model.predict_log_proba(X_encoded),
        columns=sklearn_model.classes_,
        index=X.index,
    )

    assert river_jll.columns.tolist() == sklearn_jll.columns.tolist()

    assert np.allclose(
        np.exp(river_jll.to_numpy() - river_jll.max(axis=1).to_numpy()[:, None]),
        np.exp(sklearn_jll.to_numpy() - sklearn_jll.max(axis=1).to_numpy()[:, None]),
        atol=1e-10,
    )
