# DynamicQuantizer

Adapted version of the Quantizer Observer (QO)[^/dev/api/tree/splitter/DynamicQuantizer/#1] that is applied to Stochastic Gradient Trees (SGT).

This feature quantizer starts by partitioning the inputs using the passed `radius` value. As more splits are created in the SGTs, new feature quantizers will use `std * std_prop` as the quantization radius. In the expression, `std` represents the standard deviation of the input data, which is calculated incrementally.

## Parameters

- **radius**

  *Type* → `float`

  *Default* → `0.5`

  The initial quantization radius.
- **std\_prop**

  *Type* → `float`

  *Default* → `0.25`

  The proportion of the standard deviation that is going to be used to define the radius value for new quantizer instances following the initial one.

## Methods

> [!NOTE]
>
> **update**

[^/dev/api/tree/splitter/DynamicQuantizer/#1]: Mastelini, S.M. and de Leon Ferreira, A.C.P., 2021. Using dynamical quantization to perform split attempts in online tree regressors. Pattern Recognition Letters.
