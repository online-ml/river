import numpy as np

from .. import utils


class SDFT(utils.Window):
    """Sliding Discrete Fourier Transform (SDFT).

    Initially, the coefficients are all equal to 0, up until enough values have been seen. A call
    to `numpy.fft.fft` is triggered once ``window_size`` values have been seen. Subsequent values
    will update the coefficients online. This is much faster than recomputing an FFT from scratch
    for every new value.

    Parameters:
        window_size (int): The size of the window.

    Attributes:
        fft (numpy array of complex numbers): The Fourier components.

    Example:

        ::

            >>> X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

            >>> window_size = 5
            >>> sdft = SDFT(window_size)

            >>> for i, x in enumerate(X):
            ...     sdft = sdft.update(x)
            ...
            ...     if i + 1 >= window_size:
            ...         assert np.allclose(
            ...             sdft.fft,
            ...             np.fft.fft(X[i+1 - window_size:i+1])
            ...         )

    """

    def __init__(self, window_size):
        super().__init__(window_size=window_size)
        self.fft = np.zeros(window_size)

    def update(self, x):

        # Simply append the new value if the window isn't full yet
        if len(self) < self.window_size - 1:
            self.append(x)

        # Compute an initial FFT the first time the window is full
        elif len(self) == self.window_size - 1:
            self.append(x)
            self.fft = np.fft.fft(self).tolist()

        # Update the coefficients for subsequent values
        else:
            diff = x - self[0]
            for i in range(self.window_size):
                self.fft[i] = (self.fft[i] + diff) * np.exp(2j * np.pi * i / self.window_size)
            self.append(x)

        return self
