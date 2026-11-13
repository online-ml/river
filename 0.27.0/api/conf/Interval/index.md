# Interval

An object to represent a (prediction) interval.

Users are not expected to use this class as-is. Instead, they should use the `with_interval` parameter of the `predict_one` method of any regressor or classifier wrapped with a conformal prediction method.

## Parameters

- **lower**

  *Type* → `float`

  The lower bound of the interval.
- **upper**

  *Type* → `float`

  The upper bound of the interval.

## Attributes

- **center**

  The center of the interval.
- **width**

  The width of the interval.
