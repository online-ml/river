
from river import compat
from torch import nn, optim


def build_torch_linear_regressor(n_features):
    net = nn.Sequential(
        nn.Linear(n_features,1)
    )
    return net

if __name__ == '__main__':



    model = compat.PyTorch2RiverRegressor(
        build_fn= build_torch_linear_regressor,
        loss_fn=nn.MSELoss,
        optimizer_fn=optim.SGD,
    )
    #%%

    from river.utils import check_estimator

    check_estimator(model=model)
    #%%