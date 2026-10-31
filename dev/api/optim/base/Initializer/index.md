# Initializer

An initializer is used to set initial weights in a model.

## Methods

> [!NOTE]
>
> ****call****
>
> Returns a fresh set of weights.
>
> The return type is shape-dependent (a scalar when `shape == 1`, otherwise an array), and these values are routinely used as `collections.defaultdict` factories elsewhere, so the return is intentionally left dynamic.
>
> **Parameters**
>
> - **shape** — `int` — defaults to `1`
