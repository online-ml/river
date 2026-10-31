# BinaryFocalLoss

Binary focal loss.

This implements the "star" algorithm from the appendix of the focal loss paper.

## Parameters

- **gamma**

  *Type* → `float`

  *Default* → `2`
- **beta**

  *Type* → `float`

  *Default* → `1`

## Methods

> [!NOTE]
>
> ****call****
>
> Returns the loss.
>
> **Parameters**
>
> - **y\_true** — `typing.Any`
> - **y\_pred** — `typing.Any`
>
> **Returns**
>
> `typing.Any`: The loss(es).

> [!NOTE]
>
> **gradient**
>
> Return the gradient with respect to y\_pred.
>
> **Parameters**
>
> - **y\_true** — `typing.Any`
> - **y\_pred** — `typing.Any`
>
> **Returns**
>
> `typing.Any`: The gradient(s).

> [!NOTE]
>
> **mean\_func**
>
> Mean function.
>
> This is the inverse of the link function. Typically, a loss function takes as input the raw output of a model. In the case of classification, the raw output would be logits. The mean function can be used to convert the raw output into a value that makes sense to the user, such as a probability.
>
> **Parameters**
>
> - **y\_pred** — `typing.Any`
>
> **Returns**
>
> `typing.Any`: The adjusted prediction(s).

## References

- 1. [Lin, T.Y., Goyal, P., Girshick, R., He, K. and Dollár, P., 2017. Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision (pp. 2980-2988)](<https://arxiv.org/pdf/1708.02002.pdf>)
