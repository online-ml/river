import math

from .knn import NearestNeighbours


def test_find_nearest_manhattan():

    X = [
        {'x': 0, 'y': 0},
        {'x': 0, 'y': 1},
        {'x': 1, 'y': 0},
        {'x': 1, 'y': 1},
    ]

    nn = NearestNeighbours(window_size=len(X), p=1)

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

    nn = NearestNeighbours(window_size=len(X), p=2)

    for x in X:
        nn.update(x, None)

    neighbors = nn.find_nearest(x={'x': 0, 'y': .4}, k=2)

    assert neighbors[0][0] == X[0]
    assert math.isclose(neighbors[0][2], .16)
    assert neighbors[1][0] == X[1]
    assert math.isclose(neighbors[1][2], .36)
