from river import datasets
from river import evaluate
from river import linear_model
from river import metrics
from river import optim
from river import preprocessing


def test_issue_1443():
    dataset = datasets.Phishing()

    model = preprocessing.StandardScaler() | linear_model.LogisticRegression(
        optimizer=optim.SGD(0.1)
    )

    metric = metrics.ConfusionMatrix()

    for _ in evaluate.iter_progressive_val_score(dataset, model, metric):
        pass
