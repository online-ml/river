import pandas as pd

from river import datasets, linear_model, metrics, multiclass, preprocessing, stream


def test_online_batch_consistent():

    # Batch

    batch = preprocessing.StandardScaler() | multiclass.OneVsRestClassifier(
        linear_model.LogisticRegression()
    )

    dataset = datasets.ImageSegments()

    batch_metric = metrics.MacroF1()

    for i, x in enumerate(pd.read_csv(dataset.path, chunksize=1)):
        y = x.pop("category")
        y_pred = batch.predict_many(x)
        batch.learn_many(x, y)

        for yt, yp in zip(y, y_pred):
            if yp is not None:
                batch_metric.update(yt, yp)

        if i == 30:
            break

    # Online

    online = preprocessing.StandardScaler() | multiclass.OneVsRestClassifier(
        linear_model.LogisticRegression()
    )

    online_metric = metrics.MacroF1()

    X = pd.read_csv(dataset.path)
    Y = X.pop("category")

    for i, (x, y) in enumerate(stream.iter_pandas(X, Y)):
        y_pred = online.predict_one(x)
        online.learn_one(x, y)

        if y_pred is not None:
            online_metric.update(y, y_pred)

        if i == 30:
            break

    assert online_metric.get() == batch_metric.get()
