import numpy as np

from . import window


class SDFT(window.Window):
    """Sliding Discrete Fourier Transform (SDFT).

    Initially, the coefficients are all equal to 0, up until enough values have been seen. A call
    to `numpy.fft.fft` is triggered once `window_size` values have been seen. Subsequent values
    will update the coefficients online. This is much faster than recomputing an FFT from scratch
    for every new value.

    Parameters
    ----------
    window_size
        The size of the window.

    Attributes
    ----------
    window : utils.Window
        The window of values.

    Examples
    --------

    >>> from river import utils

    >>> X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    >>> window_size = 5
    >>> sdft = utils.SDFT(window_size)

    >>> for i, x in enumerate(X):
    ...     sdft = sdft.update(x)
    ...
    ...     if i + 1 >= window_size:
    ...         assert np.allclose(sdft, np.fft.fft(X[i+1 - window_size:i+1]))

    References
    ----------
    [^1]: `Jacobsen, E. and Lyons, R., 2003. The sliding DFT. IEEE Signal Processing Magazine, 20(2), pp.74-80. <https://www.comm.utoronto.ca/~dimitris/ece431/slidingdft.pdf>`_
    [^2]: `Understanding and Implementing the Sliding DFT <https://www.dsprelated.com/showarticle/776.php>`_

    """

    def __init__(self, window_size):
        super().__init__(size=window_size)
        self.window = window.Window(size=window_size)

    def update(self, x):

        # Simply append the new value if the window isn't full yet
        if len(self.window) < self.window.size - 1:
            self.window.append(x)

        # Compute an initial FFT the first time the window is full
        elif len(self.window) == self.window.size - 1:
            self.window.append(x)
            self.extend(np.fft.fft(self.window))

        # Update the coefficients for subsequent values
        else:
            diff = x - self.window[0]
            for i in range(self.size):
                self[i] = (self[i] + diff) * np.exp(2j * np.pi * i / self.size)
            self.window.append(x)

        return self
