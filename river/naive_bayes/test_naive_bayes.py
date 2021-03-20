import pandas as pd
import pytest
from sklearn import naive_bayes as sk_naive_bayes

from river import compose, feature_extraction, naive_bayes


def yield_dataset():
    """"Incremental dataset."""
    yield from [
        ("Chinese Beijing Chinese", "yes"),
        ("Chinese Chinese Shanghai", "yes"),
        ("Chinese Macao", "yes"),
        ("Tokyo Japan Chinese", "no"),
    ]


def yield_batch_dataset():
    """"Batch dataset."""
    for x, y in yield_dataset():
        yield pd.Series([x]), pd.Series([y])


def yield_unseen_data():
    yield from [
        "Chinese Beijing Chinese",
        "Chinese Chinese Shanghai",
        "Chinese Macao",
        "Tokyo Japan Chinese",
        "new unseen data",
        "Taiwanese Taipei",
        "Chinese Shanghai" "Shanghai",
        "Chinese",
        "Tokyo Macao",
        "Tokyo Tokyo",
        "Macao Macao new",
        "new",
    ]


def yield_batch_unseen_data():
    yield from [pd.Series(x) for x in yield_unseen_data()]


@pytest.mark.parametrize(
    "inc_model, batch_model, bag, sk_model",
    [
        pytest.param(
            compose.Pipeline(
                ("tokenize", feature_extraction.BagOfWords(lowercase=False)),
                ("model", model(alpha=alpha)),
            ),
            compose.Pipeline(
                ("tokenize", feature_extraction.BagOfWords(lowercase=False)),
                ("model", model(alpha=alpha)),
            ),
            feature_extraction.BagOfWords(lowercase=False),
            sk_model(alpha=alpha),
            id=f"{model.__name__} - {alpha}",
        )
        for model, sk_model in [
            (naive_bayes.MultinomialNB, sk_naive_bayes.MultinomialNB),
            (naive_bayes.BernoulliNB, sk_naive_bayes.BernoulliNB),
            (naive_bayes.ComplementNB, sk_naive_bayes.ComplementNB),
        ]
        for alpha in [alpha for alpha in range(1, 4)]
    ],
)
def test_inc_vs_batch(inc_model, batch_model, bag, sk_model):
    """
    Assert incremental models give the same results as batch models.
    Assert river's naive_bayes module produce the same results as sklearn's naive_bayes module.

    """
    assert inc_model.predict_proba_one("not fitted yet") == {}
    assert inc_model.predict_one("not fitted yet") is None

    assert batch_model.predict_proba_many(
        pd.Series(["new", "unseen"], index=["river", "rocks"])
    ).equals(pd.DataFrame(index=["river", "rocks"]))

    assert batch_model.predict_many(
        pd.Series(["new", "unseen"], index=["river", "rocks"])
    ).equals(pd.DataFrame(index=["river", "rocks"]))

    for x, y in yield_dataset():
        inc_model = inc_model.learn_one(x, y)

    for x, y in yield_batch_dataset():
        batch_model = batch_model.learn_many(x, y)

    X = pd.concat([x for x, _ in yield_batch_dataset()])
    y = pd.concat([y for _, y in yield_batch_dataset()])

    sk_model = sk_model.fit(X=bag.transform_many(X), y=y)

    assert inc_model["model"].p_class("yes") == batch_model["model"].p_class("yes")
    assert inc_model["model"].p_class("no") == batch_model["model"].p_class("no")

    if isinstance(sk_model, sk_naive_bayes.BernoulliNB) or isinstance(
        sk_model, sk_naive_bayes.MultinomialNB
    ):

        inc_cp = inc_model["model"].p_feature_given_class
        batch_cp = batch_model["model"].p_feature_given_class

        assert inc_cp("Chinese", "yes") == batch_cp("Chinese", "yes")
        assert inc_cp("Tokyo", "yes") == batch_cp("Tokyo", "yes")
        assert inc_cp("Japan", "yes") == batch_cp("Japan", "yes")
        assert inc_cp("Chinese", "no") == batch_cp("Chinese", "no")
        assert inc_cp("Tokyo", "no") == batch_cp("Tokyo", "no")
        assert inc_cp("Japan", "no") == batch_cp("Japan", "no")
        assert inc_cp("unseen", "yes") == batch_cp("unseen", "yes")

    assert inc_model["model"].class_counts == batch_model["model"].class_counts
    assert inc_model["model"].feature_counts == batch_model["model"].feature_counts

    if isinstance(sk_model, sk_naive_bayes.ComplementNB) or isinstance(
        sk_model, sk_naive_bayes.MultinomialNB
    ):
        assert inc_model["model"].class_totals == batch_model["model"].class_totals

    if isinstance(sk_model, sk_naive_bayes.ComplementNB):
        assert inc_model["model"].feature_totals == batch_model["model"].feature_totals

    # Assert batch and incremental models give same results
    for x, x_batch in zip(yield_unseen_data(), yield_batch_unseen_data()):

        assert inc_model.predict_proba_one(x)["yes"] == pytest.approx(
            batch_model.predict_proba_many(x_batch)["yes"][0]
        )

        assert inc_model.predict_proba_one(x)["no"] == pytest.approx(
            batch_model.predict_proba_many(x_batch)["no"][0]
        )

    # Assert river produce same results as sklearn using sparse dataframe:
    for sk_preds, river_preds in zip(
        sk_model.predict_proba(bag.transform_many(X)),
        inc_model.predict_proba_many(X).values,
    ):
        for sk_pred, river_pred in zip(sk_preds, river_preds):
            assert river_pred == pytest.approx(
                1 - sk_pred
            ) or river_pred == pytest.approx(sk_pred)

    # Assert river produce same results as sklearn using dense dataframe:
    for sk_preds, river_preds in zip(
        sk_model.predict_proba(bag.transform_many(X).sparse.to_dense()),
        inc_model["model"]
        .predict_proba_many(bag.transform_many(X).sparse.to_dense())
        .values,
    ):
        for sk_pred, river_pred in zip(sk_preds, river_preds):
            assert river_pred == pytest.approx(
                1 - sk_pred
            ) or river_pred == pytest.approx(sk_pred)

    # Test class methods
    if isinstance(sk_model, sk_naive_bayes.ComplementNB) or isinstance(
        sk_model, sk_naive_bayes.MultinomialNB
    ):
        assert inc_model["model"]._more_tags() == {"positive input"}

    assert inc_model["model"]._multiclass
