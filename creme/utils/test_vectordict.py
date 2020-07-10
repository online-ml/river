from creme.utils import VectorDict


def test_vectordict():

    x = {'a': 8, 'b': -1.2, 4: 2.7}
    vx = VectorDict(x)
    assert isinstance(vx, dict)
    assert vx == x
    assert vx['a'] == 8
    assert vx[4] == 2.7
    assert vx[()] == 0
    vx[9] = 8.9
    assert vx[9] == 8.9

    y = dict()
    vy = VectorDict()
    assert vy == y

    z1 = {'a': 1, 'b': -5, 'c': -3}
    z2 = {'a': 2, 'b': 0.5, 'd': 4}
    vz1 = VectorDict(z1)
    vz2 = VectorDict(z2)
    assert vz1 == vz1
    assert +vz1 == vz1
    assert -vz1 == {'a': -1, 'b': 5, 'c': 3}
    assert vz1 + 2 == 2 + vz1 == {'a': 3, 'b': -3, 'c': -1}
    assert vz1 * 2 == 2 * vz1 == {'a': 2, 'b': -10, 'c': -6}
    assert vz1 - 2 == {'a': -1, 'b': -7, 'c': -5}
    assert vz1 / 2 == {'a': 0.5, 'b': -2.5, 'c': -1.5}
    assert vz1 + vz2 == vz2 + vz1 == vz1 + z2 == {'a': 3, 'b': -4.5, 'c': -3, 'd': 4}
    assert vz1 - vz2 == vz1 - z2 == {'a': -1, 'b': -5.5, 'c': -3, 'd': -4}
    assert vz1 @ vz2 == vz2 @ vz1 == vz1 @ z2 == -0.5

    vz1b = VectorDict(z1)
    vz1b += 2
    assert vz1b == vz1 + 2
    vz1b = VectorDict(z1)
    vz1b -= 2
    assert vz1b == vz1 - 2
    vz1b = VectorDict(z1)
    vz1b *= 2
    assert vz1b == vz1 * 2
    vz1b = VectorDict(z1)
    vz1b /= 2
    assert vz1b == vz1 / 2
    vz1b = VectorDict(z1)
    vz1b += vz2
    assert vz1b == vz1 + vz2
    vz1b = VectorDict(z1)
    vz1b += z2
    assert vz1b == vz1 + z2
    vz1b = VectorDict(z1)
    vz1b -= vz2
    assert vz1b == vz1 - vz2
    vz1b = VectorDict(z1)
    vz1b -= z2
    assert vz1b == vz1 - z2
