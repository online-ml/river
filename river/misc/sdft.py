from __future__ import annotations

import collections

import numpy as np

from river import base


class SDFT(base.Base):
    """Sliding Discrete Fourier Transform (SDFT).

    Initially, the coefficients are all equal to 0, up until enough values have been seen. A call
    to `numpy.fft.fft` is triggered once `window_size` values have been seen. Subsequent values
    will update the coefficients online. This is much faster than recomputing an FFT from scratch
    for every new value.

    Parameters
    ----------
    window_size
        The size of the window.

    Examples
    --------

    >>> import numpy as np
    >>> from river import misc

    >>> X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    >>> window_size = 5
    >>> sdft = misc.SDFT(window_size)

    >>> for i, x in enumerate(X):
    ...     sdft = sdft.update(x)
    ...
    ...     if i + 1 >= window_size:
    ...         assert np.allclose(sdft.coefficients, np.fft.fft(X[i+1 - window_size:i+1]))

    References
    ----------
    [^1]: [Jacobsen, E. and Lyons, R., 2003. The sliding DFT. IEEE Signal Processing Magazine, 20(2), pp.74-80.](https://www.comm.utoronto.ca/~dimitris/ece431/slidingdft.pdf)
    [^2]: [Understanding and Implementing the Sliding DFT](https://www.dsprelated.com/showarticle/776.php)

    """

    def __init__(self, window_size):
        self.coefficients = collections.deque(maxlen=window_size)
        self.window = collections.deque(maxlen=window_size)

    @property
    def window_size(self):
        return self.coefficients.maxlen

    def update(self, x):
        # Simply append the new value if the window isn't full yet
        if len(self.window) < self.window.maxlen - 1:
            self.window.append(x)

        # Compute an initial FFT the first time the window is full
        elif len(self.window) == self.window.maxlen - 1:
            self.window.append(x)
            self.coefficients.extend(np.fft.fft(self.window))

        # Update the coefficients for subsequent values
        else:
            diff = x - self.window[0]
            for i, c in enumerate(self.coefficients):
                self.coefficients[i] = (c + diff) * np.exp(2j * np.pi * i / self.window_size)
            self.window.append(x)

        return self
