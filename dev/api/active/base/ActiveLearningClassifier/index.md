# ActiveLearningClassifier

Base class for active learning classifiers.

## Parameters

- **classifier**

  *Type* → `base.Classifier`

  The classifier to wrap.
- **seed**

  *Type* → `int | None`

  *Default* → `None`

  Random number generator seed for reproducibility.

## Methods

> [!NOTE]
>
> **learn\_one**
>
> Update the model with a set of features `x` and a label `y`.
>
> **Parameters**
>
> - **x**
> - **y**
> - **kwargs**

> [!NOTE]
>
> **predict\_one**
>
> Predict the label of `x` and indicate whether a label is needed.
>
> **Parameters**
>
> - **x**
> - **kwargs**
>
> **Returns**
>
> The predicted label.

> [!NOTE]
>
> **predict\_proba\_one**
>
> Predict the probability of each label for `x` and indicate whether a label is needed.
>
> **Parameters**
>
> - **x**
> - **kwargs**
>
> **Returns**
>
> A dictionary that associates a probability which each label.
