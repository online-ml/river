import time

from creme import compat
from creme import compose
from creme import datasets
from creme import dummy
from creme import linear_model
from creme import metrics
from creme import optim
from creme import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier


def evaluate_model(X_y, model, metric):

    train_duration = 0
    pred_duration = 0

    for x, y in X_y:

        # Make a prediction
        tic = time.time()
        y_pred = model.predict_one(x)
        pred_duration += time.time() - tic

        # Update the metric
        metric.update(y, y_pred)

        # Update the model
        tic = time.time()
        model.fit_one(x, y)
        train_duration += time.time() - tic

    return metric, train_duration, pred_duration


models = {
    'sklearn SGDClassifier': compose.Pipeline([
        preprocessing.StandardScaler(),
        compat.CremeClassifierWrapper(
            sklearn_estimator=SGDClassifier(
                loss='log',
                learning_rate='optimal',
                fit_intercept=False
            ),
            classes=[False, True]
        )
    ]),
    'sklearn PassiveAggressiveClassifier': compose.Pipeline([
        preprocessing.StandardScaler(),
        compat.CremeClassifierWrapper(
            sklearn_estimator=PassiveAggressiveClassifier(),
            classes=[False, True]
        )
    ]),
    'No-change classifier': dummy.NoChangeClassifier(),
    'Passive-aggressive II': compose.Pipeline([
        preprocessing.StandardScaler(),
        linear_model.PAClassifier(C=1, mode=2)
    ]),
    'Logistic regression w/ VanillaSGD': compose.Pipeline([
        preprocessing.StandardScaler(),
        linear_model.LogisticRegression(
            optimizer=optim.VanillaSGD(
                lr=optim.OptimalLR()
            )
        )
    ]),
    'Logistic regression w/ Adam': compose.Pipeline([
        preprocessing.StandardScaler(),
        linear_model.LogisticRegression(optim.Adam(optim.OptimalLR()))
    ]),
    'Logistic regression w/ AdaGrad': compose.Pipeline([
        preprocessing.StandardScaler(),
        linear_model.LogisticRegression(optim.AdaGrad(optim.OptimalLR()))
    ]),
    'Logistic regression w/ RMSProp': compose.Pipeline([
        preprocessing.StandardScaler(),
        linear_model.LogisticRegression(optim.RMSProp(optim.OptimalLR()))
    ])
}

fig, ax = plt.subplots(figsize=(10, 6))

for name, model in models.items():
    print(name)
    metric, train_duration, pred_duration = evaluate_model(
        X_y=datasets.fetch_electricity(),
        model=model,
        metric=metrics.Accuracy()
    )
    print(metric)
    print(f'Training duration: {train_duration}')
    print(f'Predicting duration: {pred_duration}')
    print('-' * 10)
