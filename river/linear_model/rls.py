import numpy as np

class RLS(object):

    def __init__(self, p: int, l: float, delta: float):
        self.p = p
        self.l = l
        self.delta = delta

        self.currentStep = 0

        self.x = np.zeros((p + 1, 1))  # Column vector
        self.P = np.identity(p + 1) * self.delta

        self.estimates = []
        self.estimates.append(np.zeros((p + 1, 1)))  # Weight vector initialized to zeros

        self.Pks = []
        self.Pks.append(self.P)

    def estimate(self, xn: float, dn: float):
        # Update input vector
        self.x = np.roll(self.x, -1)
        self.x[-1, 0] = xn

        # Get previous weight vector
        wn_prev = self.estimates[-1]

        # Compute gain vector
        denominator = self.l + self.x.T @ self.Pks[-1] @ self.x
        gn = (self.Pks[-1] @ self.x) / denominator

        # Compute a priori error
        alpha = dn - (self.x.T @ wn_prev)

        # Update inverse correlation matrix
        Pn = (self.Pks[-1] - gn @ self.x.T @ self.Pks[-1]) / self.l
        self.Pks.append(Pn)

        # Update weight vector
        wn = wn_prev + gn * alpha
        self.estimates.append(wn)

        self.currentStep += 1

        return wn
