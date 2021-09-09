import datasets as datasets

from river import compat, datasets, evaluate, metrics
from torch import nn, optim


def build_torch_mlp_classifier(n_features):
    net = nn.Sequential(
        nn.Linear(n_features, 5),
        nn.Linear(5, 5),
        nn.Linear(5, 5),
        nn.Linear(5, 5),
        nn.Linear(5, 1),
        nn.Sigmoid()
    )
    return net

if __name__ == '__main__':

    model = compat.PyTorch2RiverClassifier(
        build_fn=build_torch_mlp_classifier,
        loss_fn=nn.BCELoss,
        optimizer_fn=optim.Adam,
        learning_rate=1e-3
    )

    from river.utils import check_estimator

    check_estimator(model=model)
    #dataset = datasets.Elec2()
    #metric = metrics.Accuracy()

    #print(evaluate.progressive_val_score(dataset=dataset, model= model,metric= metric).get())
