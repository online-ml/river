import pandas as pd
import random
import string
from river import linear_model, preprocessing


def test_many():

    alphabet = list(string.ascii_lowercase)
    random.seed(42)
    X = [
        {
            "c1": random.choice(alphabet),
            "c2": random.choice(alphabet),
        }
        for _ in range(4)
    ]

    oh = preprocessing.OneHotEncoder(sparse=False)
    oh = oh.learn_many(pd.DataFrame(X))
    df = oh.transform_many(pd.DataFrame(X))
    df = df.loc[:, sorted(df.columns)]

    model = linear_model.LogisticRegression()

    model.learn_many(df.iloc[:, [1, 2]], df.any(axis=1).astype(int))
    model.learn_many(df.iloc[:, [0, 1, 2]], df.any(axis=1).astype(int))
    model.learn_many(df.iloc[:, [1, 2, 0]], df.any(axis=1).astype(int))

    weights_1 = model.weights

    model = linear_model.LogisticRegression()

    model.learn_many(df.iloc[:, [1, 2, 0]], df.any(axis=1).astype(int))
    model.learn_many(df.iloc[:, [0, 1, 2]], df.any(axis=1).astype(int))
    model.learn_many(df.iloc[:, [1, 2]], df.any(axis=1).astype(int))

    weights_2 = model.weights

    assert weights_1 == weights_2
