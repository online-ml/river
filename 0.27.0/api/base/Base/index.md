# Base

Base class that is inherited by the majority of classes in River.

This base class allows us to handle the following tasks in a uniform manner:

- Getting and setting parameters
- Displaying information
- Mutating/cloning

## Methods

> [!NOTE]
>
> **clone**
>
> Return a fresh estimator with the same parameters.
>
> The clone has the same parameters but has not been updated with any data. This works by looking at the parameters from the class signature. Each parameter is either - recursively cloned if its a class. - deep-copied via `copy.deepcopy` if not. If the calling object is stochastic (i.e. it accepts a seed parameter) and has not been seeded, then the clone will not be idempotent. Indeed, this method's purpose if simply to return a new instance with the same input parameters.
>
> **Parameters**
>
> - **new\_params** — `dict[str, typing.Any] | None` — defaults to `None`
> - **include\_attributes** — `bool` — defaults to `False`

> [!NOTE]
>
> **mutate**
>
> Modify attributes.
>
> This changes parameters inplace. Although you can change attributes yourself, this is the recommended way to proceed. By default, all attributes are immutable, meaning they shouldn't be mutated. Calling `mutate` on an immutable attribute raises a `ValueError`. Mutable attributes are specified via the `_mutable_attributes` property, and are thus specified on a per-estimator basis.
>
> **Parameters**
>
> - **new\_attrs** — `dict[str, typing.Any]`
