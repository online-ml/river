from river import stats


def test_chi_squared_basic():
    chi = stats.ChiSquared()

    data = [
        ("A", 0),
        ("A", 0),
        ("B", 1),
        ("B", 1),
    ]

    for x, y in data:
        chi.update(x, y)

    assert chi.get() > 0


def test_chi_squared_zero_case():
    chi = stats.ChiSquared()

    # same class only → no dependency
    data = [
        ("A", 0),
        ("B", 0),
        ("C", 0),
    ]

    for x, y in data:
        chi.update(x, y)

    assert chi.get() == 0.0


def test_chi_squared_incremental():
    chi = stats.ChiSquared()

    chi.update("A", 0)
    first = chi.get()

    chi.update("A", 1)
    second = chi.get()

    assert second >= 0
    assert first >= 0