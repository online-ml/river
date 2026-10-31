# StaticQuantizer

Quantization strategy originally used in Stochastic Gradient Trees (SGT)[^/dev/api/tree/splitter/StaticQuantizer/#1].

Firstly, a buffer of size `warm_start` is stored. The data stored in the buffer is then used to quantize the input feature into `n_bins` intervals. These intervals will be replicated to every new quantizer. Feature values lying outside of the limits defined by the initial buffer will be mapped to the head or tail of the list of intervals.

## Parameters

- **n\_bins**

  *Type* → `int`

  *Default* → `64`

  The number of bins (intervals) to divide the input feature.
- **warm\_start**

  *Type* → `int`

  *Default* → `100`

  The number of observations used to initialize the quantization intervals.
- **buckets**

  *Type* → `list | None`

  *Default* → `None`

  This parameter is only used internally by the quantizer, so it must not be set. Once the intervals are defined, new instances of this quantizer will receive the quantization information via this parameter.

## Methods

> [!NOTE]
>
> **update**

[^/dev/api/tree/splitter/StaticQuantizer/#1]: Gouk, H., Pfahringer, B., &amp; Frank, E. (2019, October). Stochastic Gradient Trees. In Asian Conference on Machine Learning (pp. 1094-1109).
