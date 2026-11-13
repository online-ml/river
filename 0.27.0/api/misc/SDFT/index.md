# SDFT

Sliding Discrete Fourier Transform (SDFT).

Initially, the coefficients are all equal to 0, up until enough values have been seen. A call to `numpy.fft.fft` is triggered once `window_size` values have been seen. Subsequent values will update the coefficients online. This is much faster than recomputing an FFT from scratch for every new value.

## Parameters

- **window\_size**

  *Type* → `int`

  The size of the window.

## Attributes

- **window\_size**

## Examples

```python
import numpy as np
from river import misc

X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

window_size = 5
sdft = misc.SDFT(window_size)

for i, x in enumerate(X):
    sdft.update(x)
    if i + 1 >= window_size:
        assert np.allclose(sdft.coefficients, np.fft.fft(X[i+1 - window_size:i+1]))
```

## Methods

> [!NOTE]
>
> **update**

## References

- [Jacobsen, E. and Lyons, R., 2003. The sliding DFT. IEEE Signal Processing Magazine, 20(2), pp.74-80.](<https://www.comm.utoronto.ca/~dimitris/ece431/slidingdft.pdf>)
- [Understanding and Implementing the Sliding DFT](<https://www.dsprelated.com/showarticle/776.php>)
