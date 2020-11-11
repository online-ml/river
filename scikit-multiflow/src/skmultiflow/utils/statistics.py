import numpy as np

SQRTH = 7.07106781186547524401E-1
MAXLOG = 7.09782712893383996732E2


def normal_probability(a):
    """ normal_probability

    Returns the area under the Normal (Gaussian) Probability Density Function, integrated from minus infinity to
    :math:`x` (assumes mean is zero, variance is one).
    Computation is via the functions errorFunction and errorFunctionComplement.

    Parameters
    ----------
    a: The z-value

    Returns
    -------
    The probability of the z value according to the normal pdf

    """
    x = a * SQRTH
    z = np.abs(x)
    
    if z < SQRTH:
        y = 0.5 + 0.5 * error_function(x)
    else:
        y = 0.5 * error_function_complemented(z)
        if x > 0:
            y = 1.0 - y
    return y


def error_function(x):
    """ error_function

    Calculates the Error Function of the Normal Distribution

    Parameters
    ----------
    x: The x value

    Returns
    -------
    Error Function erf(x)
    """
    T = [9.60497373987051638749E0,
         9.00260197203842689217E1,
         2.23200534594684319226E3,
         7.00332514112805075473E3,
         5.55923013010394962768E4]
    U = [3.35617141647503099647E1,
         5.21357949780152679795E2,
         4.59432382970980127987E3,
         2.26290000613890934246E4,
         4.92673942608635921086E4]

    if np.abs(x) > 1.0:
        return 1.0 - error_function_complemented(x)
    else:
        z = x * x
        y = x * pol_evl(z, T, 4) / p1_evl(z, U, 5)
        return y


def error_function_complemented(a):
    """ error_function_complemented
        Calculates the complementary Error Function of the Normal Distribution.

        Parameters
        ----------
        a: The a value

        Returns
        -------
        Complementary Error Function erfc(a)
        """
    P = [2.46196981473530512524E-10,
         5.64189564831068821977E-1,
         7.46321056442269912687E0,
         4.86371970985681366614E1,
         1.96520832956077098242E2,
         5.26445194995477358631E2,
         9.34528527171957607540E2,
         1.02755188689515710272E3,
         5.57535335369399327526E2]

    Q = [1.32281951154744992508E1,
         8.67072140885989742329E1,
         3.54937778887819891062E2,
         9.75708501743205489753E2,
         1.82390916687909736289E3,
         2.24633760818710981792E3,
         1.65666309194161350182E3,
         5.57535340817727675546E2]

    R = [5.64189583547755073984E-1,
         1.27536670759978104416E0,
         5.01905042251180477414E0,
         6.16021097993053585195E0,
         7.40974269950448939160E0,
         2.97886665372100240670E0]

    S = [2.26052863220117276590E0,
         9.39603524938001434673E0,
         1.20489539808096656605E1,
         1.70814450747565897222E1,
         9.60896809063285878198E0,
         3.36907645100081516050E0]

    # if a < 0.0:
    #     x = -a
    # else:
    #     x = a
    x = np.abs(a)

    if x < 1.0:
        return 1.0 - error_function(a)

    z = -a * a

    if z < -MAXLOG:
        if a < 0:
            return 2.0
        else:
            return 0.0

    z = np.exp(z)

    if x < 8.0:
        p = pol_evl(x, P, 8)
        q = p1_evl(x, Q, 8)
    else:
        p = pol_evl(x, R, 5)
        q = p1_evl(x, S, 6)

    y = (z * p) / q

    if a < 0.0:
        y = 2.0 - y

    if y == 0.0:
        if a < 0.0:
            return 2.0
        else:
            return 0.0

    return y


def pol_evl(x, coef, N):
    """ pol_evl

    Evaluates the given polynomial of degree N at x.
    In the interest of speed, there are no checks for out of bounds arithmetic.

    Parameters
    ----------
    x: Argument of the polynomial
    coef: The coefficients of the polynomial
    N: The degree of the polynomial

    Returns
    -------

    """
    ans = coef[0]
    for i in range(1, N + 1):
        ans = ans * x + coef[i]
    return ans


def p1_evl(x, coef, N):
    """ p1_evl

    Evaluates the given polynomial of degree N at x.
    Evaluates polynomial when coefficient of N is 1.0.
    Otherwise same as polevl().
    In the interest of speed, there are no checks for out of bounds arithmetic.

    Parameters
    ----------
    x: Argument of the polynomial
    coef: The coefficients of the polynomial
    N: The degree of the polynomial

    Returns
    -------

    """
    ans = x + coef[0]
    for i in range(1, N):
        ans = ans * x + coef[i]
    return ans
