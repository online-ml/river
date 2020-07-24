import numpy as np

from creme.utils import VectorDict


def test_vectordict():

    # test empty init
    x = dict()
    vx = VectorDict()
    assert vx == x

    # test basics
    x = {'a': 8, 'b': -1.2, 4: 2.7}
    vx = VectorDict(x)
    assert vx == x
    assert vx['a'] == 8
    assert vx[4] == 2.7
    vx[9] = 8.9
    assert x[9] == vx[9] == 8.9

    # test copy
    x = {'a': 8, 'b': -1.2, 4: 2.7}
    vx = VectorDict(x, copy=True)
    assert vx == x
    vx['a'] = 2
    assert x['a'] == 8
    assert vx['a'] == 2

    # test operations
    x = {'a': 1, 'b': -5, 'c': -3}
    y = {'a': 2, 'b': 0.5, 'd': 4}
    vx = VectorDict(x)
    vy = VectorDict(y)
    assert vx == vx == x
    assert +vx == vx == x
    assert -vx == {'a': -1, 'b': 5, 'c': 3}
    assert vx + 2 == 2 + vx == {'a': 3, 'b': -3, 'c': -1}
    assert vx * 2 == 2 * vx == {'a': 2, 'b': -10, 'c': -6}
    assert vx - 2 == {'a': -1, 'b': -7, 'c': -5}
    assert vx / 2 == {'a': 0.5, 'b': -2.5, 'c': -1.5}
    assert vx + vy == vy + vx == {'a': 3, 'b': -4.5, 'c': -3, 'd': 4}
    assert vx - vy == {'a': -1, 'b': -5.5, 'c': -3, 'd': -4}
    assert vx @ vy == vy @ vx == -0.5
    vz = VectorDict(x, copy=True)
    vz += 2
    assert vz == vx + 2
    vz = VectorDict(x, copy=True)
    vz -= 2
    assert vz == vx - 2
    vz = VectorDict(x, copy=True)
    vz *= 2
    assert vz == vx * 2
    vz = VectorDict(x, copy=True)
    vz /= 2
    assert vz == vx / 2
    vz = VectorDict(x, copy=True)
    vz += vy
    assert vz == vx + vy
    vz = VectorDict(x, copy=True)
    vz -= vy
    assert vz == vx - vy

    # test default_factory
    x = {'a': 1, 'b': -5}
    y = {'b': 0.5, 'd': 4, 'e': 3, 'f': 8}
    counter = iter(range(100))
    vx = VectorDict(x, default_factory=counter.__next__)
    vy = VectorDict(y)
    assert vx @ vy == 16.5
    assert counter.__next__() == 3
    assert x['f'] == 2

    # test mask
    x = {'a': 1, 'b': -5}
    y = {'b': 0.5, 'd': 4, 'e': 3, 'f': 8}
    z = {'b': 4, 'd': 2, 'g': -1}
    vx = VectorDict(x)
    vy = VectorDict(y)
    assert vx + vy == {'a': 1, 'b': -4.5, 'd': 4, 'e': 3, 'f': 8}
    vy = VectorDict(y, mask=z)
    assert vx + vy == {'a': 1, 'b': -4.5, 'd': 4}
    vy = VectorDict(y).with_mask(z.keys())
    assert vx + vy == {'a': 1, 'b': -4.5, 'd': 4}

    # test export
    x = {'a': 1, 'b': -5}
    vx = VectorDict(x)
    nx = vx.to_numpy(['b', 'c'])
    assert isinstance(nx, np.ndarray)
    assert (vx.to_numpy(['b', 'c']) == np.array([-5, 0])).all()
