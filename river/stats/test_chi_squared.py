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

    # No dependency: all belong to same class
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

    assert first >= 0
    assert second >= 0



def test_chi_squared_exact_value():
    chi = stats.ChiSquared()

    data = [
        ("A", 0),
        ("A", 0),
        ("B", 1),
        ("B", 1),
    ]

    for x, y in data:
        chi.update(x, y)

    # Known correct value
    assert abs(chi.get() - 4.0) < 1e-6

# (SciPy validation)
def test_chi_squared_against_scipy():
    import numpy as np
    from scipy.stats import chi2_contingency

    chi = stats.ChiSquared()

    data = [
        ("A", 0),
        ("A", 0),
        ("B", 1),
        ("B", 1),
    ]

    for x, y in data:
        chi.update(x, y)

    table = np.array([
        [2, 0],
        [0, 2]
    ])

    expected, _, _, _ = chi2_contingency(table, correction=False)

    assert abs(chi.get() - expected) < 1e-6


# MULTIPLE CATEGORY TEST
def test_chi_squared_multiple_categories():
    chi = stats.ChiSquared()

    data = [
        ("A", 0),
        ("B", 1),
        ("C", 0),
        ("A", 1),
        ("B", 0),
        ("C", 1),
    ]

    for x, y in data:
        chi.update(x, y)

    assert chi.get() >= 0