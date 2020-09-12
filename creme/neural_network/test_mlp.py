from creme import neural_network as nn
from creme import optim
import numpy as np
from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing


def test_boston():

    model = nn.MLP(
        dims=(13, 20, 1),
        activations=(nn.activations.ReLU, nn.activations.Identity),
        loss=optim.losses.Squared(),
        optimizer=optim.SGD(lr=1e-3),
        seed=42
    )

    X, y = datasets.load_boston(return_X_y=True)
    X = preprocessing.scale(X)
    y = np.expand_dims(y, axis=1)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y,
        test_size=.3,
        shuffle=True,
        random_state=42
    )

    z, a = model._forward(X_train)
    model._backward(z, a, y_train)
    model.predict(X_test)
