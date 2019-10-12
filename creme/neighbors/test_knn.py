import math

from . import knn


def test_find_nearest_manhattan():

    X = [
        {'x': 0, 'y': 0},
        {'x': 0, 'y': 1},
        {'x': 1, 'y': 0},
        {'x': 1, 'y': 1},
    ]

    nn = knn.NearestNeighbours(window_size=len(X), p=1)

    for x in X:
        nn.update(x, None)

    neighbors = nn.find_nearest(x={'x': 0, 'y': .4}, k=2)

    assert neighbors[0][0] == X[0]
    assert math.isclose(neighbors[0][2], .4)
    assert neighbors[1][0] == X[1]
    assert math.isclose(neighbors[1][2], .6)


def test_find_nearest_euclidean():

    X = [
        {'x': 0, 'y': 0},
        {'x': 0, 'y': 1},
        {'x': 1, 'y': 0},
        {'x': 1, 'y': 1},
    ]

    nn = knn.NearestNeighbours(window_size=len(X), p=2)

    for x in X:
        nn.update(x, None)

    neighbors = nn.find_nearest(x={'x': 0, 'y': .4}, k=2)

    assert neighbors[0][0] == X[0]
    assert math.isclose(neighbors[0][2], .16)
    assert neighbors[1][0] == X[1]
    assert math.isclose(neighbors[1][2], .36)


def test_regression_predict():

    X_y = [
        ({'x': 0, 'y': 0}, 3),
        ({'x': 0, 'y': 1}, 4),
        ({'x': 1, 'y': 0}, 11),
        ({'x': 1, 'y': 1}, 12),
    ]

    nn = knn.KNeighborsRegressor(n_neighbors=2, window_size=len(X_y), p=1, weighted=True)

    for x, y in X_y:
        nn.update(x, y)

    # X_y[0] and X_y[1] should be the closest
    y_pred = nn.predict_one(x={'x': 0, 'y': .4})
    expected = (
        (X_y[0][1] / .4 + X_y[1][1] / .6) /
        (1 / .4 + 1 / .6)
    )
    assert math.isclose(y_pred, expected)

    # X_y[0] is the closest
    y_pred = nn.predict_one(x={'x': 0, 'y': 0})
    expected = 3
    assert math.isclose(y_pred, expected)
