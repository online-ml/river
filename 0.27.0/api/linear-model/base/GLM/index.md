# GLM

Generalized Linear Model.

This serves as a base class for linear and logistic regression.

## Parameters

- **optimizer** *(mutable)*

  *Type* → `optim.base.Optimizer`

  The sequential optimizer used for updating the weights. Note that the intercept updates are handled separately.
- **loss** *(mutable)*

  *Type* → `optim.losses.Loss`

  The loss function to optimize for.
- **l2** *(mutable)*

  *Type* → `float`

  Amount of L2 regularization used to push weights towards 0. For now, only one type of penalty can be used. The joint use of L1 and L2 is not explicitly supported.
- **l1** *(mutable)*

  *Type* → `float`

  Amount of L1 regularization used to push weights towards 0. For now, only one type of penalty can be used. The joint use of L1 and L2 is not explicitly supported.
- **intercept\_init**

  *Type* → `float`

  Initial intercept value.
- **intercept\_lr** *(mutable)*

  *Type* → `optim.base.Scheduler | float`

  Learning rate scheduler used for updating the intercept. A `optim.schedulers.Constant` is used if a `float` is provided. The intercept is not updated when this is set to 0.
- **clip\_gradient** *(mutable)*

  *Type* → `float`

  Clips the absolute value of each gradient value.
- **initializer** *(mutable)*

  *Type* → `optim.base.Initializer`

  Weights initialization scheme.

## Attributes

- **weights**

## Methods

> [!NOTE]
>
> **learn\_many**

> [!NOTE]
>
> **learn\_one**
